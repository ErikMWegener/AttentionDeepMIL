"""
create_annot_drone_bags.py

Erstellt MIL-Bags aus den punktannotierten Drohnenkacheln unter

    data/datasets/drone/2025_wheat_heads_datasets/real_data/annotated_drone_images_split_4m
        train_dataset_4m_split/train/   -> Split 'train'
        train_dataset_4m_split/val/     -> Split 'validation'
        test_dataset_4m/                -> Split 'test'

Jede Kachel (1024x1024 .tif) hat eine gleichnamige .txt mit einer Punkt-
annotation ``x y`` (Ährenmittelpunkt, Pixelkoordinaten) pro Zeile.

Ablauf (analog zu create_drone_bags.py):

  1. Über jede Kachel wird ein reguläres Gitter (patch_size/stride) gelegt.
     Ein Patch ist POSITIV, sobald mindestens ``min_points`` Punktannotationen
     in seinem Rechteck liegen, sonst NEGATIV. Es werden nur die Koordinaten
     gesammelt, keine Pixeldaten.
  2. Aus diesen beiden Patch-Pools werden pro Split zufällige Bags gemischt:
     positive Bags enthalten mindestens einen positiven Patch, negative Bags
     ausschliesslich negative Patches.
  3. Die Pixeldaten werden erst beim Schreiben geholt -- gruppiert nach
     Quellbild, damit jede Kachel nur einmal dekodiert wird.

Die Pools werden strikt pro Split gebildet, d.h. train/val/test teilen sich
kein Quellbild (kein Leakage).

Alternativ liefert ``--per_image`` den Diagnose-Modus: ein Bag pro Kachel mit
allen Gitter-Patches in Originalreihenfolge (Label = 1, sobald ein positiver
Patch enthalten ist).
"""

from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms
import dataset_manager
import argparse
import multiprocessing as mp
import os

from collections import defaultdict

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '../datasets/drone/2025_wheat_heads_datasets/real_data/'
                         'annotated_drone_images_split_4m')

# Split-Name im H5 -> Unterverzeichnis im Datensatz
SPLIT_DIRS = {
    'train': 'train_dataset_4m_split/train',
    'validation': 'train_dataset_4m_split/val',
    'test': 'test_dataset_4m',
}

IMAGE_EXTENSIONS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.JPG')


def build_transform(grayscale=True):
    if grayscale:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # In Graustufen umwandeln
            transforms.ToTensor(),                        # In einen PyTorch-Tensor umwandeln
            transforms.Normalize((0.5,), (0.5,))          # Normalisieren (wie bei MNIST üblich)
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


# ---------------------------------------------------------------------------
# Metadaten: Bilder + Punktannotationen einlesen
# ---------------------------------------------------------------------------

def load_points(txt_path):
    """Lies eine Annotationsdatei als ``(N, 2)``-Array von ``x y``-Punkten.

    Fehlende oder leere Dateien ergeben ein leeres Array -- die Kachel gilt dann
    als vollständig negativ.
    """
    if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    points = np.loadtxt(txt_path, ndmin=2)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return points[:, :2].astype(np.float64)


def load_metadata(data_root=DATA_ROOT, splits=SPLIT_DIRS):
    """Sammle pro Split die Kacheln mit ihren Punktannotationen.

    Returns ``{split: [{'image_path': str, 'image_id': str, 'points': ndarray}, ...]}``.
    """
    metadata = {}

    for split, subdir in splits.items():
        split_dir = os.path.join(data_root, subdir)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'Split-Verzeichnis nicht gefunden: {split_dir}')

        images = sorted(f for f in os.listdir(split_dir) if f.endswith(IMAGE_EXTENSIONS))
        entries = []
        for image in images:
            stem, _ = os.path.splitext(image)
            entries.append({
                'image_path': os.path.join(split_dir, image),
                'image_id': stem,
                'points': load_points(os.path.join(split_dir, f'{stem}.txt')),
            })

        num_points = sum(len(e['points']) for e in entries)
        print(f'Split {split}: {len(entries)} Kacheln, {num_points} Punktannotationen '
              f'({split_dir})')
        metadata[split] = entries

    return metadata


# ---------------------------------------------------------------------------
# Patch-Definitionen: reguläres Gitter, Label über enthaltene Punkte
# ---------------------------------------------------------------------------

def count_points_in_patch(points, coords):
    """Anzahl der Punkte im halboffenen Rechteck ``[x1, x2) x [y1, y2)``."""
    if len(points) == 0:
        return 0
    x1, y1, x2, y2 = coords
    inside = ((points[:, 0] >= x1) & (points[:, 0] < x2) &
              (points[:, 1] >= y1) & (points[:, 1] < y2))
    return int(inside.sum())


def create_patch_definitions(entries, patch_size=128, stride=128, min_points=1, desc='Patches'):
    """Lege ein reguläres Gitter über jede Kachel und labele die Patches.

    Ein Patch ist positiv, sobald mindestens ``min_points`` Punktannotationen in
    ihm liegen. Es werden nur Bildpfad, Crop-Koordinaten, Label und Punktzahl
    gespeichert -- die Pixeldaten werden erst später geholt.
    """
    positive_patches = []
    negative_patches = []

    for entry in tqdm(entries, desc=desc):
        img_w, img_h = Image.open(entry['image_path']).size
        points = entry['points']

        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                coords = (x, y, x + patch_size, y + patch_size)
                num_points = count_points_in_patch(points, coords)
                label = 1 if num_points >= min_points else 0
                patch = {'image_path': entry['image_path'], 'image_id': entry['image_id'],
                         'coords': coords, 'label': label, 'num_points': num_points}
                (positive_patches if label == 1 else negative_patches).append(patch)

    print(f'{desc}: {len(positive_patches)} positive und {len(negative_patches)} '
          f'negative Patches.')

    return positive_patches, negative_patches


# ---------------------------------------------------------------------------
# Pixeldaten holen (ein Decode pro Quellbild)
# ---------------------------------------------------------------------------

def _crop_patches_from_image(task):
    """Dekodiere ein Bild genau einmal und schneide alle gewünschten Patches aus.

    PIL dekodiert immer die komplette Datei, das Öffnen dominiert also die
    Laufzeit. Deshalb wird alles, was aus einem Bild gebraucht wird, zuerst
    gesammelt und dann in einem Rutsch geschnitten.
    """
    image_path, coords_list, grayscale = task
    patches = []
    with Image.open(image_path) as img:
        img.load()
        for coords in coords_list:
            patch = img.crop(coords)
            patch = patch.convert('L') if grayscale else patch.convert('RGB')
            patches.append(np.asarray(patch, dtype=np.uint8))
    return image_path, np.stack(patches)


def _materialize_patches(bags_instances, grayscale, workers, desc):
    """Hole die Pixeldaten für eine Liste von Bags, ein Decode pro Quellbild.

    ``bags_instances`` ist eine Liste von Instanzlisten (eine pro Bag). Zurück
    kommt eine Liste von uint8-Arrays mit Shape ``(bag_len, H, W)`` (grayscale)
    bzw. ``(bag_len, H, W, 3)``.
    """
    # image_path -> [(Bag-Index, Slot-Index, coords), ...]
    groups = defaultdict(list)
    for bag_idx, instances in enumerate(bags_instances):
        for slot_idx, instance in enumerate(instances):
            groups[instance['image_path']].append((bag_idx, slot_idx, instance['coords']))

    tasks = [(path, [c for _, _, c in entries], grayscale) for path, entries in groups.items()]
    slots = [[None] * len(instances) for instances in bags_instances]

    def scatter(result):
        image_path, patches = result
        for (bag_idx, slot_idx, _), patch in zip(groups[image_path], patches):
            slots[bag_idx][slot_idx] = patch

    if workers > 1 and len(tasks) > 1:
        with mp.Pool(min(workers, len(tasks))) as pool:
            for result in tqdm(pool.imap_unordered(_crop_patches_from_image, tasks),
                               total=len(tasks), desc=desc):
                scatter(result)
    else:
        for task in tqdm(tasks, desc=desc):
            scatter(_crop_patches_from_image(task))

    return [np.stack(bag) for bag in slots]


def _to_bag_tensor(patches):
    """Wandle einen uint8-Patch-Stapel in den normalisierten Tensor um.

    Entspricht ``ToTensor`` + ``Normalize((0.5,), (0.5,))`` aus
    :func:`build_transform`, nur auf den ganzen Bag auf einmal angewendet.
    """
    tensor = torch.from_numpy(patches)
    if tensor.ndim == 3:                       # (N, H, W) grayscale
        tensor = tensor.unsqueeze(1)
    else:                                      # (N, H, W, C) rgb
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor.float().div_(255.0).sub_(0.5).div_(0.5)


# ---------------------------------------------------------------------------
# Zufällige Bags aus den Patch-Pools
# ---------------------------------------------------------------------------

def create_bags(num_bags, mean_bag_len, var_bag_len, positive_patches, negative_patches,
                output_path, dataset_name='annot_drone_bags', split='train', bag_ratio=0.5,
                seed=0, grayscale=True, workers=None, chunk_bags=200):
    """Mische zufällige Bags aus den Patch-Definitionen und schreibe sie ins H5.

    Positive Bags enthalten eine gemischte Auswahl aus positiven und negativen
    Patches (mindestens einen positiven), negative Bags nur negative Patches.

    Erst werden die Bag-Zusammensetzungen gezogen, danach die Pixeldaten pro
    Quellbild geholt. Die Bags werden in Blöcken von ``chunk_bags`` verarbeitet,
    damit immer nur eine begrenzte Menge Patch-Daten im Speicher liegt.
    """
    if not positive_patches:
        raise ValueError(f'Split {split}: keine positiven Patches gefunden.')
    if not negative_patches:
        raise ValueError(f'Split {split}: keine negativen Patches gefunden.')

    num_pos_bags = int(num_bags * bag_ratio)
    num_neg_bags = num_bags - num_pos_bags

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    r = np.random.RandomState(seed)
    print(f'Erstelle {num_pos_bags} positive und {num_neg_bags} negative {split}-Bags '
          f'mit mittlerer Länge {mean_bag_len} und Varianz {var_bag_len}...')

    # ---- 1. Bag-Zusammensetzungen ziehen (noch keine Pixeldaten). ----------
    plans = []

    for i in range(num_pos_bags):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))
        num_positive = r.randint(1, bag_length) if bag_length > 1 else 1  # mind. ein positiver Patch
        num_negative = bag_length - num_positive

        pos_idx = r.choice(len(positive_patches), num_positive, replace=True)
        instances = [positive_patches[j] for j in pos_idx]

        if num_negative > 0:
            neg_idx = r.choice(len(negative_patches), num_negative, replace=True)
            instances += [negative_patches[j] for j in neg_idx]

        order = r.permutation(len(instances))
        instances = [instances[j] for j in order]

        plans.append({'image_id': f'{split}_{i}', 'label': 1, 'count': num_positive,
                      'instances': instances})

    for i in range(num_neg_bags):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))

        neg_idx = r.choice(len(negative_patches), bag_length, replace=True)
        instances = [negative_patches[j] for j in neg_idx]

        order = r.permutation(len(instances))
        instances = [instances[j] for j in order]

        plans.append({'image_id': f'{split}_{i+num_pos_bags}', 'label': 0, 'count': 0,
                      'instances': instances})

    # ---- 2. Pixel blockweise holen und Bags schreiben. ---------------------
    _write_plans(plans, output_path, dataset_name, split, grayscale, workers, chunk_bags)

    print(f'{num_bags} {split}-Bags fertig und in {output_path} gespeichert.')


def _write_plans(plans, output_path, dataset_name, split, grayscale, workers, chunk_bags):
    """Hole die Pixeldaten der geplanten Bags blockweise und schreibe sie ins H5."""
    writer = dataset_manager.DatasetWriter(output_path)
    num_chunks = (len(plans) + chunk_bags - 1) // chunk_bags

    for chunk_nr, start in enumerate(range(0, len(plans), chunk_bags), start=1):
        chunk = plans[start:start + chunk_bags]
        bags = _materialize_patches([p['instances'] for p in chunk], grayscale, workers,
                                    desc=f'Schneide {split} Block {chunk_nr}/{num_chunks}')

        for plan, patches in tqdm(list(zip(chunk, bags)),
                                  desc=f'Schreibe {split} Block {chunk_nr}/{num_chunks}'):
            instance_labels = [instance['label'] for instance in plan['instances']]
            point_count = sum(instance['num_points'] for instance in plan['instances'])
            writer.write(dataset_name,
                         plan['image_id'],
                         None,
                         label=plan['label'],
                         patches=_to_bag_tensor(patches),
                         count=plan['count'],
                         instance_label=torch.tensor(instance_labels),
                         split=split,
                         point_count=point_count)


# ---------------------------------------------------------------------------
# Diagnose-Modus: ein Bag pro Kachel
# ---------------------------------------------------------------------------

def create_image_bags(entries, output_path, dataset_name, split, patch_size=128, stride=128,
                      min_points=1, grayscale=True, workers=None, chunk_bags=20):
    """Ein Bag pro Kachel: alle Gitter-Patches in Originalreihenfolge.

    Das Bag-Label ist 1, sobald mindestens ein positiver Patch enthalten ist.
    Da praktisch jede annotierte Kachel Ähren enthält, entstehen hier fast nur
    positive Bags -- der Modus ist zur Inspektion/Evaluation gedacht, nicht als
    Trainingsdatensatz.
    """
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    plans = []
    for entry in tqdm(entries, desc=f'Patch-Definitionen {split} (pro Bild)'):
        img_w, img_h = Image.open(entry['image_path']).size
        points = entry['points']

        instances = []
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                coords = (x, y, x + patch_size, y + patch_size)
                num_points = count_points_in_patch(points, coords)
                instances.append({'image_path': entry['image_path'],
                                  'image_id': entry['image_id'],
                                  'coords': coords,
                                  'label': 1 if num_points >= min_points else 0,
                                  'num_points': num_points})

        num_positive = sum(instance['label'] for instance in instances)
        plans.append({'image_id': entry['image_id'],
                      'label': 1 if num_positive > 0 else 0,
                      'count': num_positive,
                      'instances': instances})

    _write_plans(plans, output_path, dataset_name, split, grayscale, workers, chunk_bags)

    num_pos_bags = sum(p['label'] for p in plans)
    print(f'{len(plans)} {split}-Bags (pro Kachel) geschrieben, davon {num_pos_bags} positiv.')


# ---------------------------------------------------------------------------

def run_bags(args, metadata):
    num_bags = dict(zip(('train', 'validation', 'test'), args.num_bags))

    for split, entries in metadata.items():
        positive_patches, negative_patches = create_patch_definitions(
            entries, patch_size=args.patch_size, stride=args.stride,
            min_points=args.min_points, desc=f'Patch-Definitionen {split}')

        create_bags(num_bags[split], args.mean_bag_len, args.var_bag_len,
                    positive_patches, negative_patches,
                    args.output_path, args.dataset_name, split=split,
                    bag_ratio=args.bag_ratio, seed=args.seed, grayscale=args.grayscale,
                    workers=args.workers, chunk_bags=args.chunk_bags)


def run_per_image(args, metadata):
    for split, entries in metadata.items():
        create_image_bags(entries, args.output_path, args.dataset_name, split,
                          patch_size=args.patch_size, stride=args.stride,
                          min_points=args.min_points, grayscale=args.grayscale,
                          workers=args.workers, chunk_bags=args.chunk_bags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Erstellt Bags aus den punktannotierten Drohnenkacheln (4 m).')
    parser.add_argument('--output_path', type=str, default='../datasets/bags/annot_drone_bags.h5',
                        help='Pfad zur H5-Datei, in die die Bags geschrieben werden.')
    parser.add_argument('--dataset_name', type=str, default='annot_drone_bags',
                        help='Name des Datensatzes innerhalb der H5-Datei.')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT,
                        help='Wurzelverzeichnis der annotierten Kacheln.')
    parser.add_argument('--num_bags', nargs='+', type=int, default=[1000, 200, 200],
                        help='Anzahl der Bags je Split (train, validation, test).')
    parser.add_argument('--patch_size', type=int, default=128, help='Kantenlänge der Patches.')
    parser.add_argument('--stride', type=int, default=None,
                        help='Schrittweite des Gitters (Default: patch_size, also überlappungsfrei).')
    parser.add_argument('--min_points', type=int, default=1,
                        help='Ab wie vielen Punktannotationen ein Patch als positiv gilt.')
    parser.add_argument('--grayscale', action='store_true', help='Patches in Graustufen speichern.')
    parser.add_argument('--mean_bag_len', type=int, default=100,
                        help='Mittlere Anzahl Instanzen pro Bag.')
    parser.add_argument('--var_bag_len', type=int, default=10,
                        help='Varianz der Instanzanzahl pro Bag.')
    parser.add_argument('--bag_ratio', type=float, default=0.5,
                        help='Anteil positiver Bags.')
    parser.add_argument('--seed', type=int, default=0, help='Random-Seed für die Reproduzierbarkeit.')
    parser.add_argument('--per_image', action='store_true',
                        help='Diagnose-Modus: ein Bag pro Kachel statt zufällig gemischter Bags.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Prozesse zum Dekodieren/Schneiden der Bilder (Default: CPU-Anzahl - 1).')
    parser.add_argument('--chunk_bags', type=int, default=200,
                        help='Anzahl Bags, deren Patches gleichzeitig im Speicher gehalten werden.')

    args = parser.parse_args()
    if args.stride is None:
        args.stride = args.patch_size

    metadata = load_metadata(args.data_root)

    if args.per_image:
        run_per_image(args, metadata)
    else:
        run_bags(args, metadata)

    print('All done!')
