"""Thesis figure: patch count -> ear count on the full images, three patch sizes.

Quantile variant 0.80/0.10, hard count at the per-model threshold calibrated on the
validation bags (exp 31, runs *_calthr). Per patch size the seed with the median
linear-conversion MAE. Data: logged per-image counts (full_image_results.json).

Usage (from the repository root):
    python eval/scripts/plot_fullimage_conversion.py
Writes eval/results/drone_fullimage_conversion.{pdf,png}.
"""
import json, os, sqlite3, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RUNS = os.path.join(ROOT, 'runs')
sys.path.insert(0, RUNS)
from full_image_count import fit_calibration

INK, INK2, GRID, AXIS = '#0b0b0b', '#52514e', '#e6e5e1', '#8a8985'
C_TEST, C_CAL = '#2a78d6', '#eb6834'          # categorical slots 1 and 2

con = sqlite3.connect(os.path.join(RUNS, 'mlflow.db'))
panels = []
for ps in (32, 64, 128):
    runs = con.execute("select run_uuid, name from runs where experiment_id=31 and lifecycle_stage='active' "
                       "and name like ?", (f'clamqu_{ps}_%_calthr',)).fetchall()
    maes = []
    for u, n in runs:
        mae = con.execute("select value from latest_metrics where run_uuid=? and key='test_linear_hard_count_counting_mae'", (u,)).fetchone()[0]
        maes.append((mae, u, n))
    maes.sort()
    mae, u, n = maes[1]                        # median of three seeds
    j = json.load(open(os.path.join(RUNS, 'mlruns', '31', u, 'artifacts', 'full_image_results.json')))
    t = pd.DataFrame(j['data'], columns=j['columns'])
    cal, test = t.iloc[:20], t.iloc[20:]
    f = fit_calibration(cal.hard_count, cal.gt_count, kind='linear')
    npatch = int(t.n_patches.iloc[0])
    rho = stats.spearmanr(test.hard_count, test.gt_count).statistic
    panels.append(dict(ps=ps, seed=n.split('_')[2], mae=mae, rho=rho, cal=cal, test=test, f=f, npatch=npatch))

plt.rcParams.update({'font.size': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2})
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.55), sharey=True, constrained_layout=True)
for ax, p in zip(axes, panels):
    n = p['npatch']
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.scatter(p['test'].hard_count / n, p['test'].gt_count, s=20, c=C_TEST, edgecolors='white', linewidths=0.5, zorder=3)
    ax.scatter(p['cal'].hard_count / n, p['cal'].gt_count, s=24, marker='^', facecolors='none', edgecolors=C_CAL, linewidths=1.1, zorder=4)
    xs = np.linspace(0, 1, 200)
    ax.plot(xs, p['f'](xs * n), color=INK, lw=1.4, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(-20, 700)
    ax.set_title(f"{p['ps']} px, {n} patches", loc='left', fontsize=8.5, color=INK)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '0.5', '1'])
    ax.text(0.03, 0.97, f"$\\rho_S$ = {p['rho']:.2f}\nMAE = {p['mae']:.1f}", transform=ax.transAxes, va='top', ha='left', fontsize=7.5, color=INK2)
axes[0].set_ylabel('Annotated ears per image')
fig.supxlabel('Share of patches counted as positive per image', fontsize=8, color=INK2)
handles = [Line2D([], [], ls='', marker='o', color=C_TEST, markeredgecolor='white', ms=5, label='Test images (70)'),
           Line2D([], [], ls='', marker='^', markerfacecolor='none', markeredgecolor=C_CAL, ms=5.5, label='Calibration images (20)'),
           Line2D([], [], color=INK, lw=1.4, label='Linear conversion, fitted on calibration images')]
fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False, fontsize=7.5)
for ext in ('pdf', 'png'):
    os.makedirs(os.path.join(ROOT, 'eval', 'results'), exist_ok=True)
    fig.savefig(os.path.join(ROOT, 'eval', 'results', f'drone_fullimage_conversion.{ext}'), dpi=200)
print([(p['ps'], p['seed'], round(p['mae'], 2), round(p['rho'], 3)) for p in panels])
