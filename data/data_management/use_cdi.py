import cv2, crop_drone_images as cdi
import matplotlib.pyplot as plt

bgr = cv2.imread("../datasets/drone/2026_wheat_heads_datasets/2026-06-05_QLB/DJI_202606051216_156_ZeroSpikeFlightQLB4m20260521/DJI_20260605121957_0002_D_point1.JPG")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)      # imread liefert BGR!

crop, bbox = cdi.crop_to_green(rgb, margin_frac=0.001, margin_px=2, open_ksize=20)
vis = cdi.draw_bbox(rgb, bbox)

cv2.imwrite("../datasets/drone/cropped/positive/crop.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

plt.imshow(crop)
plt.axis("off")
plt.show()