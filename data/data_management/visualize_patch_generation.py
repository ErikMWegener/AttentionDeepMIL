import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

IMAGE_CSV_PATH = '../datasets/gwhd_2021/competition_train.csv'
IMAGES_PATH = '../datasets/gwhd_2021/images'

def extract_box_patches(bbox, img_w, img_h, patch_size=28, stride=28, threshold=1.0):
    x1, y1, x2, y2 = bbox

    box_h = y2 - y1
    box_w = x2 - x1

    center_x = x1 + box_w // 2
    center_y = y1 + box_h // 2

    num_patch_in_width = max(1, int(box_w * (1 + 1 - threshold) / (stride)))
    num_patch_in_height = max(1, int(box_h * (1 + 1 - threshold) / (stride)))

    start_x = max(0, center_x - (num_patch_in_width * stride) // 2)
    start_y = max(0, center_y - (num_patch_in_height * stride) // 2)
    
    if start_x + num_patch_in_width * stride > img_w:
        start_x = img_w - num_patch_in_width * stride - 1
    if start_y + num_patch_in_height * stride > img_h:
        start_y = img_h - num_patch_in_height * stride - 1
    coords = []

    for y in range(min(start_y, img_h - patch_size - 1), min(img_h - 1, start_y + num_patch_in_height * stride), stride):
        if y + patch_size > img_h:
            break
        for x in range(min(start_x, img_w - patch_size - 1), min(img_w - 1, start_x + num_patch_in_width * stride), stride):
            if x + patch_size > img_w:
                break
            patch_coords = (x, y, x + patch_size, y + patch_size)
            
            coords.append(patch_coords)

    return coords

def visualize_patch_generation(image_name, stride, patch_size, threshold=0.5, dense=False):
    df = pd.read_csv(IMAGE_CSV_PATH)
    image_row = df[df['image_name'] == image_name]

    print(image_row['BoxesString'])
    if image_row.empty:
        print(f"No data found for image: {image_name}")
        return
    
    boxes = []
    for s in image_row.iloc[0]['BoxesString'].split(';'):
                if s == 'no_box':
                    continue
                if s.strip():
                    x1, y1, x2, y2 = map(int, s.split(' '))
                    boxes.append((x1, y1, x2, y2))
    
    image_path = os.path.join(IMAGES_PATH, image_name)

    with Image.open(image_path) as img:
        img_w, img_h = img.size

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                patch_coords = (x, y, x + patch_size, y + patch_size)

                patch_label = 0
                for bbox in boxes:
                    x1 = max(patch_coords[0], bbox[0])
                    y1 = max(patch_coords[1], bbox[1])
                    x2 = min(patch_coords[2], bbox[2])
                    y2 = min(patch_coords[3], bbox[3])

                    if x1 < x2 and y1 < y2:
                        if dense:
                            patch_label = 1
                            break
                        intersection = (x2 - x1) * (y2 -y1)
                        patch_area = (patch_coords[2] - patch_coords[0]) * (patch_coords[3] - patch_coords[1])
                        if intersection / patch_area > threshold:
                            patch_label = 1
                            break   
                
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
        plt.savefig(f'../../eval/results/patch_generation_{image_name}_overlap{threshold}{"" if not dense else "_dense"}.png')  # Speichern als Bilddatei
        plt.show()

visualize_patch_generation('a2a15938845d9812de03bd44799c4b1bf856a8ad11752e81c94dc8d138515021.png', stride=64, patch_size=64, threshold=1.0, dense=True)