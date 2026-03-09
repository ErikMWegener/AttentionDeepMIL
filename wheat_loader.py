"""Pytorch dataset that loads Global Wheat Head Detection dataset as bags of patches.

Each image is divided into a grid of patches. Patches overlapping with
wheat head bounding boxes are labeled positive. All patches from one
image form a bag.

The dataset can be loaded from a local directory containing a CSV file
with annotations and an image directory, or from the HuggingFace
``datasets`` library.
"""

import ast
import csv
import os

import numpy as np
import torch
import torch.utils.data as data_utils
from PIL import Image
from torchvision import transforms


class WheatHeadBags(data_utils.Dataset):
    """Loads the Global Wheat Head Detection dataset and creates bags of
    image patches.

    Each image is divided into a grid of square patches. Patches whose area
    overlaps sufficiently with at least one wheat head bounding box are
    labeled positive.  All patches extracted from a single image form one
    bag.  The bag-level label is 1 when the image contains at least one
    wheat head and 0 otherwise.

    The dataset is loaded from a local directory (Kaggle format) or, if
    unavailable, from the HuggingFace ``datasets`` library.

    Args:
        data_dir: Path to the dataset directory.  Expected layout::

            data_dir/
                train.csv       # image_id, width, height, bbox, source
                train/          # <image_id>.jpg files

        patch_size: Side length of the square patches to extract
            (default: 28, matching the existing model input).
        stride: Step size between consecutive patches.  Defaults to
            *patch_size* (non-overlapping patches).
        num_bag: Maximum number of bags (images) to use.  ``None`` means
            use every available image.
        seed: Random seed for reproducibility.
        train: If ``True`` the training split is used; otherwise the test
            split.
        overlap_threshold: Minimum fraction of the patch area that must
            overlap with a bounding box for the patch to be labeled
            positive (default: 0.25).
        train_split: Fraction of images assigned to the training set
            (default: 0.8).
        max_patches_per_bag: When set, at most this many patches are
            kept per bag (selected at random).  ``None`` keeps all
            patches.
    """

    def __init__(self, data_dir='../datasets/global-wheat-detection',
                 patch_size=28, stride=None, num_bag=None, seed=1,
                 train=True, overlap_threshold=0.25, train_split=0.8,
                 max_patches_per_bag=None):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.num_bag = num_bag
        self.train = train
        self.overlap_threshold = overlap_threshold
        self.train_split = train_split
        self.max_patches_per_bag = max_patches_per_bag

        self.r = np.random.RandomState(seed)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load dataset --------------------------------------------------
        self.images, self.annotations = self._load_data()

        # Split into train / test ----------------------------------------
        all_image_ids = sorted(self.annotations.keys())
        self.r.shuffle(all_image_ids)
        split_idx = int(len(all_image_ids) * self.train_split)

        if self.train:
            self.image_ids = all_image_ids[:split_idx]
        else:
            self.image_ids = all_image_ids[split_idx:]

        if self.num_bag is not None and self.num_bag < len(self.image_ids):
            self.image_ids = self.image_ids[:self.num_bag]

        # Pre-compute bags ----------------------------------------------
        self.bags_list, self.labels_list, self.counts_list = \
            self._create_bags()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Try local directory first, then HuggingFace."""
        if os.path.isdir(self.data_dir):
            return self._load_from_local()
        return self._load_from_huggingface()

    def _load_from_local(self):
        """Load from the Kaggle-format local directory."""
        csv_path = os.path.join(self.data_dir, 'train.csv')
        img_dir = os.path.join(self.data_dir, 'train')

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Annotation CSV not found at {csv_path}. "
                f"Expected 'train.csv' in {self.data_dir}."
            )
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Image directory not found at {img_dir}. "
                f"Expected 'train/' directory in {self.data_dir}."
            )

        images = {}
        annotations: dict[str, list] = {}

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['image_id']
                img_path = os.path.join(img_dir, f'{image_id}.jpg')
                if not os.path.isfile(img_path):
                    continue

                images[image_id] = img_path

                if image_id not in annotations:
                    annotations[image_id] = []

                bbox_str = row.get('bbox', '')
                if bbox_str:
                    bbox = ast.literal_eval(bbox_str)
                    annotations[image_id].append(bbox)

        return images, annotations

    def _load_from_huggingface(self):
        """Attempt to load from HuggingFace ``datasets``."""
        try:
            from datasets import load_dataset  # noqa: F811
        except ImportError:
            raise ImportError(
                "Could not load the dataset.  Either provide a local "
                f"directory at '{self.data_dir}' with 'train.csv' and a "
                "'train/' image directory, or install the 'datasets' "
                "package: pip install datasets"
            )

        try:
            ds = load_dataset('Global-Wheat', split='train')
        except Exception as e:
            raise RuntimeError(
                f"Could not load 'Global-Wheat' from HuggingFace: {e}\n"
                f"Please provide the dataset locally at '{self.data_dir}' "
                "with the following structure:\n"
                f"  {self.data_dir}/train.csv\n"
                f"  {self.data_dir}/train/<image_id>.jpg"
            ) from e

        images = {}
        annotations: dict[str, list] = {}

        for item in ds:
            image_id = str(item['image_id'])
            images[image_id] = item['image']

            bboxes: list = []
            if 'objects' in item and 'bbox' in item['objects']:
                bboxes = list(item['objects']['bbox'])
            elif 'bbox' in item:
                raw = item['bbox']
                bbox = (ast.literal_eval(raw)
                        if isinstance(raw, str) else raw)
                bboxes.append(bbox)

            annotations[image_id] = bboxes

        return images, annotations

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overlap(patch_box, bbox):
        """Return the fraction of the patch area covered by *bbox*.

        Args:
            patch_box: ``(x_min, y_min, x_max, y_max)``
            bbox: ``[x_min, y_min, width, height]``
        """
        px1, py1, px2, py2 = patch_box
        bx1, by1, bw, bh = bbox
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(px1, bx1)
        iy1 = max(py1, by1)
        ix2 = min(px2, bx2)
        iy2 = min(py2, by2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        patch_area = (px2 - px1) * (py2 - py1)
        return intersection / patch_area if patch_area > 0 else 0.0

    def _get_image(self, image_id):
        """Return a PIL ``Image`` in RGB mode."""
        img_data = self.images[image_id]
        if isinstance(img_data, str):
            return Image.open(img_data).convert('RGB')
        return img_data.convert('RGB')

    def _extract_patches(self, image_id):
        """Extract labelled patches from one image.

        Returns:
            patches: list of PIL Image crops
            patch_labels: list of 0/1 ints
            wheat_head_count: total wheat heads in the image
        """
        img = self._get_image(image_id)
        bboxes = self.annotations[image_id]
        img_w, img_h = img.size

        patches = []
        patch_labels = []

        for y in range(0, img_h - self.patch_size + 1, self.stride):
            for x in range(0, img_w - self.patch_size + 1, self.stride):
                patch_box = (x, y, x + self.patch_size, y + self.patch_size)
                patch = img.crop(patch_box)

                has_wheat = any(
                    self._compute_overlap(patch_box, bb)
                    >= self.overlap_threshold
                    for bb in bboxes
                )
                patches.append(patch)
                patch_labels.append(1 if has_wheat else 0)

        return patches, patch_labels, len(bboxes)

    # ------------------------------------------------------------------
    # Bag creation
    # ------------------------------------------------------------------

    def _create_bags(self):
        """Build all bags, labels, and wheat-head counts."""
        bags_list = []
        labels_list = []
        counts_list = []

        for image_id in self.image_ids:
            patches, patch_labels, wh_count = \
                self._extract_patches(image_id)
            if not patches:
                continue

            # Optionally sub-sample patches
            if (self.max_patches_per_bag is not None
                    and len(patches) > self.max_patches_per_bag):
                idx = self.r.choice(
                    len(patches), self.max_patches_per_bag, replace=False)
                patches = [patches[i] for i in idx]
                patch_labels = [patch_labels[i] for i in idx]

            bag = torch.stack([self.transform(p) for p in patches])
            instance_labels = torch.tensor(patch_labels, dtype=torch.long)

            bags_list.append(bag)
            labels_list.append(instance_labels)
            counts_list.append(wh_count)

        return bags_list, labels_list, counts_list

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.bags_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        instance_labels = self.labels_list[index]
        bag_label = max(instance_labels)
        wheat_head_count = self.counts_list[index]

        return bag, [bag_label, instance_labels,
                     torch.tensor([wheat_head_count])]
