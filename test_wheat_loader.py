"""Tests for wheat_loader.py WheatHeadBags dataset."""

import csv
import os

import numpy as np
import torch
from PIL import Image

from wheat_loader import WheatHeadBags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_dataset(tmp_path, num_images=5, img_size=56,
                              bboxes_per_image=2, bbox_size=10):
    """Create a small synthetic dataset in the Kaggle CSV format.

    Returns the path to the dataset directory.
    """
    data_dir = str(tmp_path / 'wheat')
    img_dir = os.path.join(data_dir, 'train')
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'train.csv')
    rows = []
    rng = np.random.RandomState(0)
    for i in range(num_images):
        image_id = f'img_{i:04d}'
        img = Image.fromarray(
            rng.randint(0, 255, (img_size, img_size, 3)).astype(np.uint8))
        img.save(os.path.join(img_dir, f'{image_id}.jpg'))

        for j in range(bboxes_per_image):
            x = int(rng.randint(0, img_size - bbox_size))
            y = int(rng.randint(0, img_size - bbox_size))
            bbox = [x, y, bbox_size, bbox_size]
            rows.append({
                'image_id': image_id,
                'width': img_size,
                'height': img_size,
                'bbox': str(bbox),
                'source': 'synthetic',
            })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['image_id', 'width', 'height', 'bbox', 'source'])
        writer.writeheader()
        writer.writerows(rows)

    return data_dir


def _create_dataset_no_wheat(tmp_path, num_images=3, img_size=56):
    """Create a dataset where no image has wheat head annotations."""
    data_dir = str(tmp_path / 'wheat_empty')
    img_dir = os.path.join(data_dir, 'train')
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'train.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['image_id', 'width', 'height', 'bbox', 'source'])
        writer.writeheader()
        for i in range(num_images):
            image_id = f'empty_{i:04d}'
            img = Image.fromarray(
                np.random.randint(0, 255, (img_size, img_size, 3),
                                  dtype=np.uint8))
            img.save(os.path.join(img_dir, f'{image_id}.jpg'))
            # Write row with empty bbox
            writer.writerow({
                'image_id': image_id,
                'width': img_size,
                'height': img_size,
                'bbox': '',
                'source': 'synthetic',
            })

    return data_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWheatHeadBags:
    """Core tests for WheatHeadBags dataset."""

    def test_bag_shape_and_types(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=5,
                                             img_size=56)
        ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=42,
                           train=True, train_split=0.8)

        assert len(ds) > 0

        bag, label = ds[0]
        # bag: (N, 1, 28, 28)
        assert bag.ndim == 4
        assert bag.shape[1] == 1  # grayscale
        assert bag.shape[2] == 28
        assert bag.shape[3] == 28

        # label: [bag_label, instance_labels, wheat_head_count]
        assert len(label) == 3
        bag_label, instance_labels, wh_count = label
        assert bag_label.dim() == 0 or bag_label.numel() == 1
        assert instance_labels.shape[0] == bag.shape[0]
        assert wh_count.numel() == 1

    def test_train_test_split(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=10,
                                             img_size=56)
        train_ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=1,
                                 train=True, train_split=0.8)
        test_ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=1,
                                train=False, train_split=0.8)

        assert len(train_ds) + len(test_ds) == 10

    def test_num_bag_limits_images(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=10,
                                             img_size=56)
        ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=1,
                           train=True, train_split=1.0, num_bag=3)

        assert len(ds) == 3

    def test_max_patches_per_bag(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=3,
                                             img_size=84)
        ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=1,
                           train=True, train_split=1.0,
                           max_patches_per_bag=4)

        for i in range(len(ds)):
            bag, _ = ds[i]
            assert bag.shape[0] <= 4

    def test_instance_labels_are_binary(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=5,
                                             img_size=56)
        ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=42,
                           train=True, train_split=1.0)

        for i in range(len(ds)):
            _, label = ds[i]
            instance_labels = label[1]
            unique = torch.unique(instance_labels)
            for v in unique:
                assert v.item() in (0, 1)

    def test_wheat_head_count_nonnegative(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=5,
                                             img_size=56,
                                             bboxes_per_image=3)
        ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=42,
                           train=True, train_split=1.0)

        for i in range(len(ds)):
            _, label = ds[i]
            wh_count = label[2].item()
            assert wh_count >= 0

    def test_bag_label_consistency(self, tmp_path):
        """bag_label should be 1 iff any instance label is 1."""
        data_dir = _create_synthetic_dataset(tmp_path, num_images=5,
                                             img_size=56)
        ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=42,
                           train=True, train_split=1.0)

        for i in range(len(ds)):
            _, label = ds[i]
            bag_label = int(label[0].item())
            has_positive = int(label[1].sum().item() > 0)
            assert bag_label == has_positive

    def test_different_patch_sizes(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=3,
                                             img_size=112)
        for ps in [14, 28, 56]:
            ds = WheatHeadBags(data_dir=data_dir, patch_size=ps, seed=1,
                               train=True, train_split=1.0)
            bag, _ = ds[0]
            assert bag.shape[2] == ps
            assert bag.shape[3] == ps

    def test_stride_parameter(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=2,
                                             img_size=56)
        ds_no_overlap = WheatHeadBags(data_dir=data_dir, patch_size=28,
                                      stride=28, seed=1,
                                      train=True, train_split=1.0)
        ds_overlap = WheatHeadBags(data_dir=data_dir, patch_size=28,
                                   stride=14, seed=1,
                                   train=True, train_split=1.0)

        # Overlapping stride should produce more patches per image
        bag_no, _ = ds_no_overlap[0]
        bag_ov, _ = ds_overlap[0]
        assert bag_ov.shape[0] >= bag_no.shape[0]

    def test_reproducibility_with_same_seed(self, tmp_path):
        data_dir = _create_synthetic_dataset(tmp_path, num_images=5,
                                             img_size=56)
        ds1 = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=7,
                            train=True, train_split=0.8)
        ds2 = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=7,
                            train=True, train_split=0.8)

        assert len(ds1) == len(ds2)
        for i in range(len(ds1)):
            b1, l1 = ds1[i]
            b2, l2 = ds2[i]
            assert torch.equal(b1, b2)
            assert torch.equal(l1[1], l2[1])


class TestComputeOverlap:
    """Tests for the static _compute_overlap helper."""

    def test_no_overlap(self):
        assert WheatHeadBags._compute_overlap(
            (0, 0, 10, 10), [20, 20, 10, 10]) == 0.0

    def test_full_overlap(self):
        assert WheatHeadBags._compute_overlap(
            (0, 0, 10, 10), [0, 0, 10, 10]) == 1.0

    def test_partial_overlap(self):
        overlap = WheatHeadBags._compute_overlap(
            (0, 0, 10, 10), [5, 5, 10, 10])
        assert 0.0 < overlap < 1.0
        # Intersection: (5,5)-(10,10) = 5x5=25, patch area 100
        assert abs(overlap - 0.25) < 1e-9

    def test_bbox_inside_patch(self):
        overlap = WheatHeadBags._compute_overlap(
            (0, 0, 100, 100), [10, 10, 20, 20])
        # Intersection: 20*20=400, patch area 10000
        assert abs(overlap - 0.04) < 1e-9


class TestDataLoaderCompatibility:
    """Verify that WheatHeadBags works with DataLoader (batch_size=1)."""

    def test_dataloader_iteration(self, tmp_path):
        import torch.utils.data as data_utils

        data_dir = _create_synthetic_dataset(tmp_path, num_images=4,
                                             img_size=56)
        ds = WheatHeadBags(data_dir=data_dir, patch_size=28, seed=1,
                           train=True, train_split=1.0)
        loader = data_utils.DataLoader(ds, batch_size=1, shuffle=False)

        for batch_idx, (data, label) in enumerate(loader):
            bag_label = label[0]
            instance_labels = label[1]
            assert data.ndim == 5  # (batch, N, C, H, W)
            assert data.shape[0] == 1
            assert bag_label.ndim == 1 or bag_label.numel() == 1


class TestMissingDataset:
    """Verify helpful errors when the dataset is not available."""

    def test_missing_csv(self, tmp_path):
        data_dir = str(tmp_path / 'bad')
        img_dir = os.path.join(data_dir, 'train')
        os.makedirs(img_dir, exist_ok=True)
        # No train.csv
        try:
            WheatHeadBags(data_dir=data_dir, patch_size=28)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_missing_image_dir(self, tmp_path):
        data_dir = str(tmp_path / 'bad2')
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, 'train.csv')
        with open(csv_path, 'w') as f:
            f.write('image_id,width,height,bbox,source\n')
        try:
            WheatHeadBags(data_dir=data_dir, patch_size=28)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
