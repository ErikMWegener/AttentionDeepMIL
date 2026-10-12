"""Thesis figure: true vs. predicted positive-patch counts on the high-density MNIST bags.

Redraws the two `aggregated_counting_plot.png` artifacts of the `density_check` runs
(MLflow experiment `Default`, softmax and sigmoid attention, five seeds) with larger
lettering for the two-column subfigure layout of the thesis (each panel 0.48 textwidth).
Same encoding as the original artifact: one colour per seed, red dashed line for perfect
predictions. Both panels share the axis limits so they can be compared directly.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RUNS = {
    'softmax': '0cfb1c6149b540b3979637888e351d85',
    'sigmoid': 'e40e8a67a58a4defa9ca62f8726f901b',
}
SEED_PLOT_COLORS = ['red', 'blue', 'green', 'orange', 'purple']   # wie runs/main_mlflow.py
LIM = (-4, 104)

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'xtick.labelsize': 12.5,
                     'ytick.labelsize': 12.5, 'legend.fontsize': 12})


def load(run_id):
    path = os.path.join(ROOT, 'runs', 'mlruns', '0', run_id, 'artifacts', 'aggregated_run_results.json')
    d = json.load(open(path))
    col = {c: i for i, c in enumerate(d['columns'])}
    return [(r[col['seeds']], r[col['count_truth']], r[col['count_pred']]) for r in d['data']]


for name, run_id in RUNS.items():
    rows = load(run_id)
    seeds = sorted({s for s, _, _ in rows})

    fig, ax = plt.subplots(figsize=(4.4, 4.9), constrained_layout=True)
    for i, seed in enumerate(seeds):
        t = [r[1] for r in rows if r[0] == seed]
        p = [r[2] for r in rows if r[0] == seed]
        ax.scatter(t, p, color=SEED_PLOT_COLORS[i % len(SEED_PLOT_COLORS)], label=f'Seed {seed}',
                   alpha=0.6, edgecolors='w', linewidths=0.5, s=30)
    ax.plot([0, 100], [0, 100], 'r--', lw=1.5, label='Perfect prediction')

    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect('equal')
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel('True count')
    ax.set_ylabel('Predicted count')
    fig.legend(loc='outside upper center', ncol=3, frameon=False, handlelength=1.2,
               handletextpad=0.4, columnspacing=1.0)

    os.makedirs(os.path.join(ROOT, 'eval', 'results'), exist_ok=True)
    out = os.path.join(ROOT, 'eval', 'results', f'mnist_{name}_scatter.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print('geschrieben:', out, f'({len(rows)} Bags, Seeds {seeds})')
