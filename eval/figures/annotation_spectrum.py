"""Abbildung 'Spectrum of Annotation' (fig:annot_spek).

Zwei Modi:
  --scan    durchsucht GWHD nach gut lesbaren Bildausschnitten und legt einen
            Kontaktbogen mit den besten Kandidaten ab
  --render  zeichnet die vier Annotationsstufen fuer einen gewaehlten Ausschnitt

Die vier Stufen werden alle aus derselben Ground Truth (den GWHD-Boxen)
abgeleitet: Punkt = Boxmittelpunkt, Count = Anzahl Boxen, Bag-Label = Count > 0.
Die Abbildung zeigt damit eine Informationsprojektion, nicht vier Datensaetze.
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

GWHD = 'data/datasets/gwhd_2021'
OUT = 'eval/results/figures'  # run from the repository root
CROP = 512


def parse_boxes(s):
    if not isinstance(s, str) or s == 'no_box':
        return np.zeros((0, 4), dtype=float)
    return np.array([[float(v) for v in b.split()] for b in s.split(';')])


def load_meta():
    parts = []
    for split in ('train', 'val', 'test'):
        df = pd.read_csv(f'{GWHD}/competition_{split}.csv')
        df['split'] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def crop_stats(boxes, x, y, size=CROP):
    """Boxen relativ zum Crop; trennt vollstaendig enthaltene von angeschnittenen."""
    if len(boxes) == 0:
        return np.zeros((0, 4)), 0
    rel = boxes - np.array([x, y, x, y])
    inside = (rel[:, 0] >= 0) & (rel[:, 1] >= 0) & (rel[:, 2] <= size) & (rel[:, 3] <= size)
    overlap = ((rel[:, 2] > 0) & (rel[:, 0] < size) & (rel[:, 3] > 0) & (rel[:, 1] < size))
    return rel[inside], int((overlap & ~inside).sum())


def score_crop(img, boxes, x, y, size=CROP, target=14):
    full, n_cut = crop_stats(boxes, x, y, size)
    n = len(full)
    if n < 6 or n > 26:
        return None
    patch = img[y:y + size, x:x + size]
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    s_count = np.exp(-0.5 * ((n - target) / 4.0) ** 2)
    s_cut = np.exp(-0.10 * n_cut)
    # Schaerfe: gut sichtbare Aehrenkontur statt Bewegungsunschaerfe
    s_sharp = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 400.0, 1.0)
    # Objektgroesse: im 512er Crop sollen Aehren ca. 50-120 px lang sein
    w = np.median(np.maximum(full[:, 2] - full[:, 0], full[:, 3] - full[:, 1]))
    s_size = np.exp(-0.5 * ((w - 85.0) / 35.0) ** 2)
    # Streuung: Objekte ueber den Ausschnitt verteilt, nicht in einer Ecke
    cx, cy = (full[:, 0] + full[:, 2]) / 2, (full[:, 1] + full[:, 3]) / 2
    s_spread = min((cx.std() + cy.std()) / (0.55 * size), 1.0)

    total = s_count * s_cut * (0.35 + 0.65 * s_sharp) * (0.4 + 0.6 * s_size) * (0.4 + 0.6 * s_spread)
    return dict(score=total, n=n, n_cut=n_cut, med_size=w, sharp=s_sharp)


def scan(args):
    meta = load_meta()
    if args.domains:
        meta = meta[meta.domain.isin(args.domains.split(','))]
    meta = meta.sample(n=min(args.n_images, len(meta)), random_state=args.seed)

    rows = []
    for i, r in enumerate(meta.itertuples(), 1):
        path = f'{GWHD}/images/{r.image_name}'
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        boxes = parse_boxes(r.BoxesString)
        if len(boxes) == 0:
            continue
        for y in range(0, img.shape[0] - CROP + 1, args.stride):
            for x in range(0, img.shape[1] - CROP + 1, args.stride):
                st = score_crop(img, boxes, x, y)
                if st:
                    rows.append(dict(image_name=r.image_name, domain=r.domain, x=x, y=y, **st))
        if i % 50 == 0:
            print(f'  {i}/{len(meta)} Bilder, {len(rows)} Kandidaten')

    cand = pd.DataFrame(rows).sort_values('score', ascending=False)
    # pro Quellbild nur den besten Ausschnitt, damit der Bogen Vielfalt zeigt
    cand = cand.drop_duplicates('image_name').head(args.top).reset_index(drop=True)
    cand.to_csv(f'{OUT}/crop_candidates.csv', index=False)

    ncol, nrow = 4, int(np.ceil(len(cand) / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow))
    for ax, r in zip(axes.ravel(), cand.itertuples()):
        img = cv2.cvtColor(cv2.imread(f'{GWHD}/images/{r.image_name}'), cv2.COLOR_BGR2RGB)
        full, _ = crop_stats(parse_boxes(meta.set_index('image_name').BoxesString[r.image_name]), r.x, r.y)
        ax.imshow(img[r.y:r.y + CROP, r.x:r.x + CROP])
        for b in full:
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, ec='#ffd400', lw=1.2))
        ax.set_title(f'[{r.Index}] {r.domain}\nn={r.n} score={r.score:.2f}', fontsize=8)
        ax.axis('off')
    for ax in axes.ravel()[len(cand):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{OUT}/crop_candidates.png', dpi=110)
    print(f'\n{len(cand)} Kandidaten -> {OUT}/crop_candidates.png')
    print(cand[['image_name', 'domain', 'x', 'y', 'n', 'n_cut', 'score']].to_string())


ACCENT = '#ffd400'
EDGE = '#1a1a1a'


def _badge(ax, text, size=13):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=size, fontweight='bold', color='#111111',
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec=EDGE, lw=1.1, alpha=0.92))


def render(args):
    meta = load_meta().set_index('image_name')
    if args.index is not None:
        cand = pd.read_csv(f'{OUT}/crop_candidates.csv')
        r = cand.iloc[args.index]
        name, x, y = r.image_name, int(r.x), int(r.y)
    else:
        name, x, y = args.image, args.x, args.y

    img = cv2.cvtColor(cv2.imread(f'{GWHD}/images/{name}'), cv2.COLOR_BGR2RGB)
    patch = img[y:y + CROP, x:x + CROP]
    boxes, _ = crop_stats(parse_boxes(meta.BoxesString[name]), x, y)
    n = len(boxes)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2

    titles = ['(a) Bounding Boxes', '(b) Point Annotaion',
              '(c) Count per Bag', '(d) Bag-Label']
    subs = [f'{4 * n} coordinates', f'{2 * n} coordinates', '1 number', '1 bit']

    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.35))
    for ax, t, s in zip(axes, titles, subs):
        ax.imshow(patch)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#555555'); sp.set_linewidth(0.8)
        ax.set_title(t, fontsize=11, pad=6)
        ax.set_xlabel(s, fontsize=9.5, color='#444444', labelpad=4)

    for b in boxes:
        axes[0].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                    fill=False, ec=ACCENT, lw=1.6))
    axes[1].scatter(cx, cy, s=48, marker='o', c=ACCENT, edgecolors=EDGE, linewidths=1.0, zorder=3)
    _badge(axes[2], f'count = {n}')
    _badge(axes[3], 'Wheat heads present\nyes  (y = 1)', size=11)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{args.out}.{ext}', dpi=300, bbox_inches='tight')
    print(f'{OUT}/{args.out}.pdf  ({name} @ {x},{y}, n={n})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)

    s = sub.add_parser('scan')
    s.add_argument('--n_images', type=int, default=400)
    s.add_argument('--stride', type=int, default=128)
    s.add_argument('--top', type=int, default=12)
    s.add_argument('--domains', default=None, help='kommagetrennt, z.B. ETHZ_1,Arvalis_2')
    s.add_argument('--seed', type=int, default=0)
    s.set_defaults(func=scan)

    r = sub.add_parser('render')
    r.add_argument('--index', type=int, default=None, help='Zeile aus crop_candidates.csv')
    r.add_argument('--image', default=None)
    r.add_argument('--x', type=int, default=0)
    r.add_argument('--y', type=int, default=0)
    r.add_argument('--out', default='annotation_spectrum')
    r.set_defaults(func=render)

    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    a.func(a)
