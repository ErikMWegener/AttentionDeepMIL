"""Abbildung 'Counting difficulties' (fig:count_diff_ex).

Drei Stufen, damit nicht tausende Ausschnitte per Auge gesichtet werden muessen:

  index   einmaliger Durchlauf ueber eine domaenen-stratifizierte Stichprobe;
          bewertet jeden Ausschnitt nach vier Schwierigkeitsmassen -> CSV
  sheet   Kontaktbogen der Top-Kandidaten eines Masses (visuelle Endauswahl)
  render  finale Abbildung aus vier per Index gewaehlten Ausschnitten

Die Masse sind Proxys, keine Wahrheit. Sie heben die Trefferquote so weit an,
dass ein Bogen mit 12 Kandidaten reicht -- die Auswahl bleibt visuell.
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
INDEX_CSV = f'{OUT}/difficulty_index.csv'
CROP = 320


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


def max_iou_per_box(b):
    """Fuer jede Box die groesste IoU mit einer anderen Box -> mutual occlusion."""
    n = len(b)
    if n < 2:
        return np.zeros(n)
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-6)
    np.fill_diagonal(iou, 0.0)
    return iou.max(axis=1)


def index_images(args):
    meta = load_meta()
    # stratifiziert ueber Domaenen: erst dadurch deckt die Stichprobe die
    # Beleuchtungs- und Wachstumsvielfalt von GWHD ueberhaupt ab
    per = max(1, args.n_images // meta.domain.nunique())
    meta = (meta.groupby('domain', group_keys=False)
                .apply(lambda d: d.sample(min(per, len(d)), random_state=args.seed)))

    rows = []
    for i, r in enumerate(meta.itertuples(), 1):
        img = cv2.imread(f'{GWHD}/images/{r.image_name}')
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = parse_boxes(r.BoxesString)
        if len(boxes) < 4:
            continue

        val = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32) / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        ious = max_iou_per_box(boxes)
        diag = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        # Texturenergie je Box: Aehren tragen Grannen und Aehrchen, ein davor
        # liegendes Blatt ist glatt -> auffaellig niedrige Energie = verdeckt
        tex = np.array([lap[int(b[1]):int(b[3]), int(b[0]):int(b[2])].mean()
                        / (val[int(b[1]):int(b[3]), int(b[0]):int(b[2])].mean() + 0.05)
                        if b[3] > b[1] and b[2] > b[0] else 0.0 for b in boxes])
        tex_ref = np.median(tex) + 1e-6

        h, w = gray.shape
        for y in range(0, h - CROP + 1, args.stride):
            for x in range(0, w - CROP + 1, args.stride):
                sel = ((boxes[:, 0] >= x) & (boxes[:, 1] >= y) &
                       (boxes[:, 2] <= x + CROP) & (boxes[:, 3] <= y + CROP))
                n = int(sel.sum())
                if n < args.min_boxes:
                    continue
                v = val[y:y + CROP, x:x + CROP]
                sharp = float(lap[y:y + CROP, x:x + CROP].var())
                p_hi, p_lo = float((v > 0.85).mean()), float((v < 0.30).mean())
                d = diag[sel]
                rows.append(dict(
                    image_name=r.image_name, domain=r.domain, x=x, y=y, n=n,
                    # mutual occlusion: mittlere staerkste Ueberlappung je Box
                    overlap=float(ious[sel].mean()),
                    # Verdeckung durch Blaetter (Proxy ueber Texturverlust)
                    occlusion=float((tex[sel] < 0.55 * tex_ref).mean()),
                    # Groessenvarianz innerhalb eines Ausschnitts
                    scale=float(np.percentile(d, 90) / max(np.percentile(d, 10), 1e-6)),
                    # harter Schattenwurf: ueberstrahlt UND abgesoffen zugleich
                    illum=float(2 * np.sqrt(p_hi * p_lo)),
                    brightness=float(v.mean()), sharp=sharp,
                ))
        if i % 100 == 0:
            print(f'  {i}/{len(meta)} Bilder, {len(rows)} Ausschnitte')

    df = pd.DataFrame(rows)
    df.to_csv(INDEX_CSV, index=False)
    print(f'\n{len(df)} Ausschnitte -> {INDEX_CSV}')
    print(df[['overlap', 'occlusion', 'scale', 'illum', 'brightness', 'sharp']].describe().to_string())


def sheet(args):
    df = pd.read_csv(INDEX_CSV)
    meta = load_meta().set_index('image_name')
    if args.min_n:
        df = df[df.n >= args.min_n]
    df = df[(df.brightness >= args.min_bright) & (df.brightness <= args.max_bright)]
    if args.min_sharp:
        df = df[df.sharp >= df.sharp.quantile(args.min_sharp)]
    df = (df.sort_values(args.metric, ascending=False)
            .drop_duplicates('image_name').head(args.top).reset_index(drop=True))

    ncol, nrow = 4, int(np.ceil(len(df) / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.5 * nrow))
    for ax, r in zip(np.atleast_1d(axes).ravel(), df.itertuples()):
        img = cv2.cvtColor(cv2.imread(f'{GWHD}/images/{r.image_name}'), cv2.COLOR_BGR2RGB)
        ax.imshow(img[r.y:r.y + CROP, r.x:r.x + CROP])
        if args.boxes:
            b = parse_boxes(meta.BoxesString[r.image_name]) - np.array([r.x, r.y, r.x, r.y])
            for q in b[(b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 2] <= CROP) & (b[:, 3] <= CROP)]:
                ax.add_patch(Rectangle((q[0], q[1]), q[2] - q[0], q[3] - q[1],
                                       fill=False, ec='#ffd400', lw=1.2))
        ax.set_title(f'[{r.Index}] {r.domain}\n{args.metric}={getattr(r, args.metric):.2f}  n={r.n}',
                     fontsize=8)
        ax.axis('off')
    for ax in np.atleast_1d(axes).ravel()[len(df):]:
        ax.axis('off')
    fig.suptitle(f'Top candidates: {args.metric}', fontsize=12)
    fig.tight_layout()
    path = f'{OUT}/cand_{args.metric}.png'
    fig.savefig(path, dpi=110)
    df.to_csv(f'{OUT}/cand_{args.metric}.csv', index=False)
    print(f'{path}\n' + df[['image_name', 'domain', 'x', 'y', 'n', args.metric]].to_string())


LABELS = {
    1:   '(a) Overlap of ears',
    2: '(b) Occlusion by leaves',
    3:     '(c) Size variability',
    4:     '(d) Illumination',
}


def render(args):
    """--pick metric:index  (viermal, in gewuenschter Panel-Reihenfolge)"""
    meta = load_meta().set_index('image_name')
    picks = []
    number = 1
    for spec in args.pick:
        metric, idx = spec.split(':')
        r = pd.read_csv(f'{OUT}/cand_{metric}.csv').iloc[int(idx)]
        picks.append((number, r))
        number += 1

    fig, axes = plt.subplots(1, len(picks), figsize=(2.9 * len(picks), 3.3))
    for ax, (number, r) in zip(np.atleast_1d(axes), picks):
        img = cv2.cvtColor(cv2.imread(f'{GWHD}/images/{r.image_name}'), cv2.COLOR_BGR2RGB)
        ax.imshow(img[int(r.y):int(r.y) + CROP, int(r.x):int(r.x) + CROP])
        if args.boxes:
            b = parse_boxes(meta.BoxesString[r.image_name]) - np.array([r.x, r.y, r.x, r.y])
            for q in b[(b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 2] <= CROP) & (b[:, 3] <= CROP)]:
                ax.add_patch(Rectangle((q[0], q[1]), q[2] - q[0], q[3] - q[1],
                                       fill=False, ec='#ffd400', lw=1.4))
        ax.set_title(LABELS[number], fontsize=11, pad=6)
        ax.set_xlabel(r.domain, fontsize=9, color='#444444', labelpad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#555555'); sp.set_linewidth(0.8)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{args.out}.{ext}', dpi=300, bbox_inches='tight')
    print(f'{OUT}/{args.out}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)

    i = sub.add_parser('index')
    i.add_argument('--n_images', type=int, default=900)
    i.add_argument('--stride', type=int, default=160)
    i.add_argument('--min_boxes', type=int, default=5)
    i.add_argument('--seed', type=int, default=0)
    i.set_defaults(func=index_images)

    s = sub.add_parser('sheet')
    s.add_argument('metric', choices=list(LABELS))
    s.add_argument('--top', type=int, default=12)
    s.add_argument('--min_n', type=int, default=0)
    s.add_argument('--boxes', action='store_true')
    s.add_argument('--min_bright', type=float, default=0.0)
    s.add_argument('--max_bright', type=float, default=1.0)
    s.add_argument('--min_sharp', type=float, default=0.0, help='Quantil, z.B. 0.5')
    s.set_defaults(func=sheet)

    r = sub.add_parser('render')
    r.add_argument('--pick', nargs='+', required=True, help='z.B. overlap:0 occlusion:3')
    r.add_argument('--boxes', action='store_true')
    r.add_argument('--out', default='difficulty_examples')
    r.set_defaults(func=render)

    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    a.func(a)
