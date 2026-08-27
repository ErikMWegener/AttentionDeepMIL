"""
crop_drone_images.py

Bestimmt die Region-of-Interest (ROI) in einem Drohnenbild: sucht den groessten
zusammenhaengenden Vegetationsbereich (gruen) und liefert ein umschliessendes
Rechteck mit Rand-Toleranz zurueck.

Gedacht als Vorfilter, damit positive Patches nur innerhalb des Pflanzen-
bestands ausgeschnitten werden.

Erwartetes Eingabeformat: HxWx3, RGB, uint8 (nicht BGR!). Bei cv2.imread ist das
Bild BGR -> vorher cv2.cvtColor(..., cv2.COLOR_BGR2RGB).
"""

from __future__ import annotations
import numpy as np
import cv2


def vegetation_mask(img_rgb: np.ndarray,
                    method: str = "exg",
                    exg_thresh: float = 0.10,
                    hsv_lower=(35, 40, 40),
                    hsv_upper=(90, 255, 255)) -> np.ndarray:
    """Binaere Vegetationsmaske (uint8, 0/255).

    method="exg"      : Excess-Green-Index (2g - r - b) auf normierten Kanaelen,
                        FESTE physikalische Schwelle `exg_thresh` (roher ExG,
                        ca. -1..2; Vegetation liegt typ. > 0.1, Boden darunter).
                        Robust auch wenn der Bestand den ganzen Rahmen fuellt.
    method="exg_otsu" : ExG mit adaptiver Otsu-Schwelle. NUR sinnvoll, wenn der
                        Plot eine kleinere Region im Bodenumfeld ist. Fuellt die
                        Vegetation den Rahmen, halbiert Otsu sie faelschlich.
    method="hsv"      : klassisches Gruen-Thresholding im HSV-Raum.
    """
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, "erwarte HxWx3 RGB"

    if method in ("exg", "exg_otsu"):
        rgb = img_rgb.astype(np.float32)
        total = rgb.sum(axis=2, keepdims=True) + 1e-6
        r, g, b = (rgb / total).transpose(2, 0, 1)
        exg = 2.0 * g - r - b                       # roh, ca. [-1, 2]
        if method == "exg":
            mask = (exg > exg_thresh).astype(np.uint8) * 255
        else:
            exg_u8 = cv2.normalize(exg, None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
            _, mask = cv2.threshold(exg_u8, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "hsv":
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lower, np.uint8),
                           np.array(hsv_upper, np.uint8))
    else:
        raise ValueError(f"unbekannte method: {method!r}")

    return mask


def largest_green_bbox(img_rgb: np.ndarray,
                       method: str = "exg",
                       exg_thresh: float = 0.10,
                       open_ksize: int = 0,
                       fill_ksize: int = 0,
                       margin_frac: float = 0.05,
                       margin_px: int = 0,
                       min_area: int = 0,
                       return_mask: bool = False):
    """Bounding-Box (x0, y0, x1, y1) um den groessten gruenen Bereich, inkl. Rand.

    open_ksize  : kleine gruene Sprenkel entfernen VOR der Komponentenauswahl.
                  0 = aus (Default). Vorsicht: frisst duenne Bestandskanten
                  (Grannen) an, meist nicht noetig.
    fill_ksize  : Loecher NACH der Komponentenauswahl schliessen, nur innerhalb
                  der gewaehlten Komponente. 0 = aus. Kann NICHT mehr zu Unkraut
                  ausserhalb bruecken (im Gegensatz zu close vor der Auswahl).
    margin_frac : Rand-Toleranz als Anteil der jeweiligen Box-Seite (pro Achse).
    margin_px   : zusaetzlicher absoluter Rand in Pixeln.
    min_area    : minimale Flaeche der groessten Komponente, sonst None.

    Rueckgabe: bbox oder None, wenn keine Vegetation gefunden wird.
               (bbox ist im Slicing-Format: img[y0:y1, x0:x1])
    """
    H, W = img_rgb.shape[:2]
    mask = vegetation_mask(img_rgb, method=method, exg_thresh=exg_thresh)

    # Optional: kleine Sprenkel weg, BEVOR die groesste Komponente gesucht wird
    if open_ksize:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # groesste zusammenhaengende Komponente (Label 0 = Hintergrund).
    # Der dichte Bestand ist von sich aus EINE Komponente; verstreutes Unkraut
    # bildet getrennte, kleinere Komponenten und faellt hier automatisch weg.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return (None, mask) if return_mask else None

    idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return (None, mask) if return_mask else None

    # nur die gewaehlte Komponente behalten (Unkraut ist damit sicher raus)
    comp = (labels == idx).astype(np.uint8) * 255

    # Optional: Loecher innerhalb der Komponente fuellen (bruecken unmoeglich)
    if fill_ksize:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_ksize, fill_ksize))
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k)

    x, y, w, h = cv2.boundingRect(comp)

    # Rand-Toleranz addieren und auf Bildgrenzen begrenzen
    mx = int(round(margin_frac * w + margin_px))
    my = int(round(margin_frac * h + margin_px))
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)

    bbox = (int(x0), int(y0), int(x1), int(y1))
    return (bbox, comp) if return_mask else bbox


def crop_to_green(img_rgb: np.ndarray, **kwargs):
    """Schneidet das Bild auf die gruene ROI zu. Gibt (crop, bbox) zurueck."""
    bbox = largest_green_bbox(img_rgb, **kwargs)
    if bbox is None:
        return img_rgb, None
    x0, y0, x1, y1 = bbox
    return img_rgb[y0:y1, x0:x1], bbox


def draw_bbox(img_rgb: np.ndarray, bbox, color=(255, 0, 0), thickness=4):
    """Zeichnet die ROI zur visuellen Kontrolle ein (Kopie zurueck)."""
    out = img_rgb.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
    return out