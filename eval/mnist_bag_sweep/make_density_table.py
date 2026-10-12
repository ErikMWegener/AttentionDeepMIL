#!/usr/bin/env python
"""Erzeugt die LaTeX-Tabelle softmax vs. sigmoid aus den density_check-Laeufen.

Transponiert gegenueber dem Toolbox-Standard: Kennzahlen als Zeilen, Aktivierung
als Spalten. Bei elf Kennzahlen waere die uebliche Anordnung fuer eine
Thesis-Seite zu breit.

    python eval/mnist_bag_sweep/make_density_table.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toolbox import loading  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "mnist_density_activation.tex")

# (Metrik, Label, Nachkommastellen, besser = groesser?)
ROWS = [
    ("accuracy",                  "Bag accuracy",              3, True),
    ("auc",                       "Bag AUC",                   3, True),
    ("patch_level_auc",           "Patch AUC (unthresholded)", 3, True),
    ("patch_level_auc_converted", "Patch AUC (thresholded)",   3, True),
    ("patch_level_precision",     "Patch precision",           3, True),
    ("patch_level_recall",        "Patch recall",              3, True),
    ("count_mae",                 "Count MAE",                 2, False),
    ("count_bias",                "Count bias",                2, None),   # naeher an 0
    ("count_r2",                  "Count $R^2$",               3, True),
]


def _num(value, prec):
    """Echtes Minuszeichen statt Bindestrich -- sonst setzt LaTeX einen Trennstrich."""
    text = f"{value:.{prec}f}"
    return text.replace("-", "$-$")


def cell(series, prec):
    s = series.dropna()
    if not len(s):
        return "--"
    return (f"{_num(s.median(), prec)} "
            f"[{_num(s.quantile(.25), prec)}, {_num(s.quantile(.75), prec)}]")


def main():
    df = loading.load_runs("runs/mlflow.db", ["Default"], level="child", status="all")
    df = loading.apply_filters(df, ["parent_run_name=density_check"])
    groups = {a: g for a, g in df.groupby("attention_activation")}
    n = {a: len(g) for a, g in groups.items()}

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Attention activation on the high-density MNIST bags "
        r"(\texttt{mnist\_bags\_even}, mean bag length 100, positive-instance "
        r"count drawn uniformly over the bag length). Median over five seeds, "
        r"interquartile range in brackets. Bold marks the better median; given "
        r"the seed spread of the sigmoid runs it does not denote a significant "
        r"difference.}",
        r"  \label{tab:mnist-density-activation}",
        r"  \small",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        f"    Metric & softmax ($n={n.get('softmax', 0)}$) "
        f"& sigmoid ($n={n.get('sigmoid', 0)}$) \\\\",
        r"    \midrule",
    ]

    for key, label, prec, bigger_better in ROWS:
        vals = {}
        for act in ("softmax", "sigmoid"):
            g = groups.get(act)
            vals[act] = cell(g[key], prec) if g is not None and key in g else "--"
            
        # Welche Spalte wird hervorgehoben?
        med = {}
        for act in ("softmax", "sigmoid"):
            g = groups.get(act)
            s = g[key].dropna() if g is not None and key in g else None
            med[act] = s.median() if s is not None and len(s) else None
        best = None
        if None not in med.values() and med["softmax"] != med["sigmoid"]:
            if bigger_better is None:                       # Bias: naeher an null
                best = min(med, key=lambda a: abs(med[a]))
            elif bigger_better:
                best = max(med, key=lambda a: med[a])
            else:
                best = min(med, key=lambda a: med[a])
        for act in ("softmax", "sigmoid"):
            if act == best:
                vals[act] = r"\textbf{" + vals[act] + "}"

        lines.append(f"    {label} & {vals['softmax']} & {vals['sigmoid']} \\\\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    path = os.path.normpath(OUT)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("geschrieben:", path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
