import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

IMAGE_CSV_PATH = '../datasets/gwhd_2021/competition_train.csv'
GWHD_IMAGES_PATH = '../datasets/gwhd_2021/images'
IMAGES_PATH = '/home/erik/AttentionDeepMIL/data/datasets/drone/2025_wheat_heads_datasets/synthetic_data/datasets/wheat_dataset_4m_MIDDLE_130_70'


def load_boxes(image_name):
    """Bounding Boxes einer GWHD-Kachel aus der Competition-CSV."""
    df = pd.read_csv(IMAGE_CSV_PATH)
    image_row = df[df['image_name'] == image_name]
    if image_row.empty:
        print(f"No data found for image: {image_name}")
        return None

    boxes = []
    for s in image_row.iloc[0]['BoxesString'].split(';'):
        if s == 'no_box' or not s.strip():
            continue
        x1, y1, x2, y2 = map(int, s.split(' '))
        boxes.append((x1, y1, x2, y2))
    return boxes


def load_points(image_path):
    """Punktannotationen aus der gleichnamigen .txt (je Zeile 'x y')."""
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    if not os.path.exists(txt_path):
        return None

    points = []
    with open(txt_path) as f:
        for line in f:
            if line.strip():
                x, y = line.split()[:2]
                points.append((float(x), float(y)))
    return np.array(points, dtype=float).reshape(-1, 2)


def count_points_in_patch(points, patch_coords):
    """Halboffenes Intervall wie in create_annot_drone_bags.py: ein Punkt auf
    einer Gitterkante zaehlt genau einmal, fuer den Nachbarpatch nicht mehr."""
    x1, y1, x2, y2 = patch_coords
    inside = ((points[:, 0] >= x1) & (points[:, 0] < x2) &
              (points[:, 1] >= y1) & (points[:, 1] < y2))
    return int(inside.sum())


def label_from_boxes(patch_coords, boxes, threshold, dense=False):
    """Patch ist positiv, sobald eine Box mehr als `threshold` seiner Flaeche
    abdeckt; mit `dense` genuegt jede Beruehrung."""
    patch_area = ((patch_coords[2] - patch_coords[0]) *
                  (patch_coords[3] - patch_coords[1]))
    for bbox in boxes:
        x1 = max(patch_coords[0], bbox[0])
        y1 = max(patch_coords[1], bbox[1])
        x2 = min(patch_coords[2], bbox[2])
        y2 = min(patch_coords[3], bbox[3])
        if x1 < x2 and y1 < y2:
            if dense:
                return 1
            if (x2 - x1) * (y2 - y1) / patch_area > threshold:
                return 1
    return 0

def extract_box_patches(bbox, img_w, img_h, patch_size=28, stride=28, threshold=1.0):
    x1, y1, x2, y2 = bbox

    box_h = y2 - y1
    box_w = x2 - x1

    center_x = x1 + box_w // 2
    center_y = y1 + box_h // 2

    # num_patch_in_width = max(1, int(box_w * (1 + 1 - threshold) / (stride)))
    # num_patch_in_height = max(1, int(box_h * (1 + 1 - threshold) / (stride)))

    # start_x = max(0, center_x - (num_patch_in_width * stride) // 2)
    # start_y = max(0, center_y - (num_patch_in_height * stride) // 2)
    
    # if start_x + num_patch_in_width * stride > img_w:
    #     start_x = img_w - num_patch_in_width * stride - 1
    # if start_y + num_patch_in_height * stride > img_h:
    #     start_y = img_h - num_patch_in_height * stride - 1
    # coords = []

    # for y in range(min(start_y, img_h - patch_size - 1), min(img_h - 1, start_y + num_patch_in_height * stride), stride):
    #     if y + patch_size > img_h:
    #         break
    #     for x in range(min(start_x, img_w - patch_size - 1), min(img_w - 1, start_x + num_patch_in_width * stride), stride):
    #         if x + patch_size > img_w:
    #             break
    #         patch_coords = (x, y, x + patch_size, y + patch_size)
            
    #         coords.append(patch_coords)
    coords = []
    x = max(0, center_x - patch_size // 2)
    y = max(0, center_y - patch_size // 2)
    coords.append((x, y, x + patch_size, y + patch_size))
    return coords

def visualize_patch_generation(image_name, stride, patch_size, threshold=0.5, dense=False, gwhd=False):
    boxes = []
    if gwhd:
        boxes = load_boxes(image_name)
        if boxes is None:
            return

    image_path = os.path.join(GWHD_IMAGES_PATH if gwhd else IMAGES_PATH, image_name)

    with Image.open(image_path) as img:
        img_w, img_h = img.size

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                patch_coords = (x, y, x + patch_size, y + patch_size)
                patch_label = 0
                if gwhd:
                    patch_label = label_from_boxes(patch_coords, boxes, threshold, dense)
                    if dense and patch_label == 1:
                        continue

                edgecolor = 'red' if patch_label == 1 else 'lightgray'
                linewidth = 2.5 if patch_label == 1 else 0.5
                alpha = 1.0 if patch_label == 1 else 0.5

                rect = patches.Rectangle(
                    (x, y), patch_size, patch_size, 
                    linewidth=linewidth, edgecolor=edgecolor, facecolor='none', alpha=alpha
                )
                ax.add_patch(rect)
        if dense:

            for bbox in boxes:
                dense_coords = extract_box_patches(bbox, img_w, img_h, patch_size, stride, threshold=threshold)

                for coords in dense_coords:
                    rect = patches.Rectangle(
                        (coords[0], coords[1]), patch_size, patch_size, 
                        linewidth=2.5, edgecolor='red', facecolor='none', alpha=1.0
                    )
                    ax.add_patch(rect)
        if gwhd:
            for bbox in boxes:
                box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                bx = patches.Rectangle(
                    (bbox[0], bbox[1]), box_w, box_h, 
                    linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--'
                )
                ax.add_patch(bx)

        plt.title(f'Patch Generation for {image_name}\nRed: Positive Patches, Blue dashed: Original BBoxes')
        plt.axis('off')
        plt.tight_layout()
        #plt.savefig(f'../../eval/results/patch_generation_{image_name}_overlap{threshold}{"" if not dense else "_dense"}.png')  # Speichern als Bilddatei
        plt.show()

def save_patches(image_name, stride, patch_size, out_dir, threshold=0.5,
                 dense=False, gwhd=False, min_points=1, max_per_class=None,
                 scale=1, seed=0):
    """Schneidet das Patch-Gitter aus und legt die Patches als PNG ab.

    Die Label-Quelle haengt am Datensatz: GWHD ueber die Bounding Boxes,
    sonst ueber die Punktannotationen der gleichnamigen .txt. Ohne beides
    gibt es kein Patch-Label und damit nichts zu trennen.

    Ergebnis: out_dir/positive/ und out_dir/negative/, Dateiname traegt
    Koordinaten und -- bei Punktannotation -- die Zahl der Punkte im Patch.
    """
    boxes, points = None, None
    image_path = os.path.join(GWHD_IMAGES_PATH if gwhd else IMAGES_PATH, image_name)

    if gwhd:
        boxes = load_boxes(image_name)
        if boxes is None:
            return
    else:
        points = load_points(image_path)
        if points is None:
            print(f"No annotation found next to {image_path} - no patch labels possible")
            return

    for sub in ('positive', 'negative'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    stem = os.path.splitext(image_name)[0]
    candidates = {'positive': [], 'negative': []}

    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img_w, img_h = img.size

        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                patch_coords = (x, y, x + patch_size, y + patch_size)
                patch_label = 0
                if gwhd:
                    n_points = None
                    patch_label = label_from_boxes(patch_coords, boxes, threshold, dense)
                    if dense and patch_label == 1:
                        continue      
                else:
                    n_points = count_points_in_patch(points, patch_coords)
                    patch_label = int(n_points >= min_points)

                name = f'{stem}_x{x}_y{y}'
                if n_points is not None:
                    name += f'_n{n_points}'
                candidates['positive' if patch_label else 'negative'].append((patch_coords, name))

        if dense:
            for bbox in boxes:
                dense_coords = extract_box_patches(bbox, img_w, img_h, patch_size, stride, threshold=threshold)
                for coords in dense_coords:
                    x,y,_,_ = coords
                    name = f'{stem}_x{x}_y{y}'
                    if n_points is not None:
                        name += f'_n{n_points}'
                    candidates['positive'].append((coords, name))



        rng = np.random.RandomState(seed)
        counts = {}
        for cls, items in candidates.items():
            if max_per_class is not None and len(items) > max_per_class:
                idx = rng.choice(len(items), max_per_class, replace=False)
                items = [items[i] for i in sorted(idx)]
            for patch_coords, name in items:
                patch = img.crop(patch_coords)
                if scale != 1:
                    # NEAREST: bei 64 px Patches soll die Skalierung fuer die
                    # Abbildung nichts weichzeichnen, was nicht da ist
                    patch = patch.resize((patch_size * scale, patch_size * scale),
                                         Image.NEAREST)
                patch.save(os.path.join(out_dir, cls, f'{name}.png'))
            counts[cls] = len(items)

    total = {c: len(v) for c, v in candidates.items()}
    print(f'{image_name}: {counts["positive"]}/{total["positive"]} positive, '
          f'{counts["negative"]}/{total["negative"]} negative -> {out_dir}')


if __name__ == '__main__':
    #visualize_patch_generation('wheat_field_000.png', stride=64, patch_size=64, threshold=1.0, dense=False, gwhd=False)
    save_patches('0ae9b1f31324a493bea1fef7dd5afe675234584d2495dbe4d43c829e6dbcbd86.png', stride=128, patch_size=128,
                 out_dir='../../tex/masterarbeit/figures/gwhd_patches', gwhd=True, threshold=1.0, dense=True, min_points=1,
                 max_per_class=12, scale=4)