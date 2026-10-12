#!/usr/bin/env python
"""Neuaggregation ohne abgestuerzte Seeds (Gutachten: Crashes nicht in die Mediane).

Ein Seed gilt als abgestuerzt, wenn er alle Test-Bags derselben Klasse zuordnet
(``toolbox.loading.mark_crashed``).  Das Skript

1. listet alle abgestuerzten Seeds aller Experimente (``crash_seeds.csv``) und
   zaehlt sie je Experiment (``crash_overview.csv``),
2. rechnet die betroffenen Thesis-Tabellen ohne diese Seeds neu und schreibt sie
   im Layout der Thesis nach ``tables_crash_excluded.tex``
   (Tab. counting-core, count-mechanism, mnist-gated, drone_synth),
3. gibt die Zahlen aus, die im Fliesstext von 6.1 und 6.5.2 stehen.

Die Abbildung zur Bag-Klassifikation entsteht mit der Toolbox, siehe
``eval/make_thesis_figures.sh``.

    python eval/scripts/crash_exclusion.py [--outdir eval/results/crash_exclusion]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from toolbox import loading  # noqa: E402

DB = os.path.normpath(os.path.join(HERE, "..", "..", "runs", "mlflow.db"))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "crash_exclusion"))
DER = r"num_bags,bag_length=dataset:bags(\d+)_len(\d+)"
SEED_ORDER = ["seed_1", "seed_10", "seed_100", "seed_1000", "seed_10000"]


def load(exp, status="FINISHED"):
    df = loading.load_runs(DB, [exp], level="child", status=status)
    if "dataset" in df.columns and df["dataset"].astype(str).str.contains("bags\\d+_len").any():
        df = loading.derive(df, [DER])
    return df


def med(s):
    s = s.dropna()
    return s.median() if len(s) else np.nan


def f(v, prec):
    return "--" if pd.isna(v) else f"{v:.{prec}f}"


def signed(v, prec):
    return "--" if pd.isna(v) else f"${v:+.{prec}f}$".replace("+-", "-")


# --------------------------------------------------------------------------- #
# 1. Crash-Liste ueber alle Experimente                                        #
# --------------------------------------------------------------------------- #
def crash_audit(outdir):
    import mlflow
    loading.connect(DB)
    exps = mlflow.MlflowClient().search_experiments()
    rows, overview = [], []
    for exp in sorted(exps, key=lambda e: int(e.experiment_id)):
        try:
            df = loading.load_runs(DB, [exp.experiment_id], level="child", status="all")
        except SystemExit:
            continue
        if "precision" not in df.columns:
            continue
        judged = df[df["precision"].notna()]
        hit = judged[judged["crashed"] == 1]
        overview.append({"experiment_id": int(exp.experiment_id), "experiment": exp.name,
                         "seeds_judged": len(judged), "crashed": len(hit)})
        for _, r in hit.iterrows():
            rows.append({"experiment_id": int(exp.experiment_id), "experiment": exp.name,
                         "parent_run_name": r.get("parent_run_name"), "seed": r["run_name"],
                         "status": r["status"],
                         **{k: r.get(k) for k in ("accuracy", "auc", "precision", "recall",
                                                  "count_mae", "count_bias",
                                                  "patch_level_auc")}})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "crash_seeds.csv"), index=False)
    ov = pd.DataFrame(overview)
    ov.to_csv(os.path.join(outdir, "crash_overview.csv"), index=False)
    print("== Crashes je Experiment (alle Status) ==")
    print(ov.to_string(index=False))


# --------------------------------------------------------------------------- #
# 2. MNIST (Abschnitt 6.1)                                                     #
# --------------------------------------------------------------------------- #
def mnist(tex):
    df = load("mnist_bag_sweep_5seeds")
    ok = df[df["crashed"] == 0]
    print("\n== MNIST, 5 Seeds ==")
    hit = df[df["crashed"] == 1]
    print(f"abgestuerzt: {len(hit)} von {len(df)} Runs; je Modell:",
          hit.groupby("model").size().to_dict(), "; je Bag-Laenge:",
          hit.groupby("bag_length").size().to_dict(), "; davon alle positiv:",
          int((hit["recall"] == 1).sum()))
    near = df[(df["accuracy"] <= 0.5) & (df["crashed"] == 0)]
    for _, r in near.iterrows():
        print(f"  Acc <= 0.5, aber kein Crash: {r['parent_run_name']} / {r['run_name']} "
              f"acc={r['accuracy']:.3f} auc={r['auc']:.3f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f} MAE={r['count_mae']:.2f} Bias={r['count_bias']:.2f}")

    # --- Abschnitt 6.1.1: Bag-AUC, Bag-Laenge 10, 500 Bags -------------------
    sub = df[(df["bag_length"] == 10) & (df["num_bags"] == 500) & (df["model"] == "attention")]
    print(f"6.1.1 AUC L10/500 attention, 5 Seeds: alle {med(sub['auc']):.3f}, "
          f"ohne Crash {med(sub[sub['crashed'] == 0]['auc']):.3f} "
          f"(n={int((sub['crashed'] == 0).sum())})")
    old = load("mnist_bag_sweep")
    sub = old[(old["bag_length"] == 10) & (old["num_bags"] == 500) & (old["model"] == "attention")]
    print(f"6.1.1 AUC L10/500 attention, 3 Seeds (exp mnist_bag_sweep): alle "
          f"{med(sub['auc']):.3f}, ohne Crash {med(sub[sub['crashed'] == 0]['auc']):.3f} "
          f"(n={int((sub['crashed'] == 0).sum())}); Seeds: "
          + ", ".join(f"{r.run_name}={r.auc:.3f}{'*' if r.crashed else ''}"
                      for r in sub.itertuples()))
    print("6.1.1 Bag-AUC-Median je Konfiguration (alle -> ohne Crash), nur wo verschieden:")
    for key, g in df.groupby(["bag_length", "num_bags", "model"]):
        a, b = med(g["auc"]), med(g[g["crashed"] == 0]["auc"])
        if g["crashed"].any():
            print(f"   {key}: {a:.3f} -> {b:.3f}  (crashed {int(g['crashed'].sum())})")

    # --- Tab. counting-core ---------------------------------------------------
    big, big_ok = df[df["num_bags"] >= 150], ok[ok["num_bags"] >= 150]
    lines = []
    for L in (10, 50, 100):
        for m in ("attention", "gated_attention"):
            g = big_ok[(big_ok["bag_length"] == L) & (big_ok["model"] == m)]
            n_cr = int(big[(big["bag_length"] == L) & (big["model"] == m)]["crashed"].sum())
            lines.append((L, m, len(g), n_cr, med(g["count_mae"]), med(g["count_bias"]),
                          med(g["count_r2"]), med(g["counting_accuracy"])))
    print("\nTab. counting-core (L, Modell, n, crashed, MAE, Bias, R2, exakt, Bias/MAE):")
    for r in lines:
        print("  ", r[:4], " ".join(f"{v:.3f}" for v in r[4:]), f"{-r[5] / r[4]:.2f}")
    body = []
    for i, (L, m, n, c, mae, bias, r2, acc) in enumerate(lines):
        if i and i % 2 == 0:
            body.append(r"					\midrule")
        body.append(f"					{L} & {m.replace('_', chr(92) + '_')} & {n} & {c} & "
                    f"{f(mae, 2)} & {f(bias, 2)} & {f(r2, 2)} & {f(acc, 2)} \\\\")
    tex.append(r"""% --- Tabelle counting-core (6.1.2), ohne abgestuerzte Seeds -----------------
			\begin{table}[htbp]
				\centering
				\small
				\begin{tabular}{clcccccc}
					\toprule
					bag\_length & model & n & crashed & count\_mae & count\_bias & count\_r2 & count\_accuracy \\
					\midrule
""" + "\n".join(body) + r"""
					\bottomrule
				\end{tabular}
				\caption[Counting performance by bag size and model variants]{Counting performance by bag size and model variant
				(including only datasets with at least 150 training bags), median over the seeds that did not crash.
				\emph{n} counts the seeds used, \emph{crashed} the excluded seeds (Section~\ref{sec:eval}).}
				\label{tab:counting-core}
			\end{table}
""")

    # --- 6.1.2: R2 und MAE ueber der Trainingsmenge, Bag-Laenge 100 -----------
    print("\n6.1.2 Bag-Laenge 100, Median ueber beide Modelle (alle -> ohne Crash):")
    for nb in sorted(df["num_bags"].unique()):
        g = df[(df["bag_length"] == 100) & (df["num_bags"] == nb)]
        go = g[g["crashed"] == 0]
        print(f"   {nb:>3} Bags: R2 {med(g['count_r2']):.3f} -> {med(go['count_r2']):.3f}, "
              f"MAE {med(g['count_mae']):.2f} -> {med(go['count_mae']):.2f} "
              f"(n {len(g)} -> {len(go)})")

    # --- Tab. count-mechanism --------------------------------------------------
    pm = big[big["patch_level_auc"].notna()]
    pm_ok = pm[pm["crashed"] == 0]
    body = []
    print("\nTab. count-mechanism (L, n, crashed, raw, thresholded, diff, P, R):")
    for i, L in enumerate((10, 50, 100)):
        g = pm_ok[pm_ok["bag_length"] == L]
        n_cr = int(big[big["bag_length"] == L]["crashed"].sum())
        raw, thr = med(g["patch_level_auc"]), med(g["patch_level_auc_converted"])
        p, r = med(g["patch_level_precision"]), med(g["patch_level_recall"])
        print(f"   {L} {len(g)} {n_cr} {raw:.3f} {thr:.3f} {raw - thr:.3f} {p:.3f} {r:.3f}")
        if i:
            body.append(r"					\midrule")
        body.append(f"					{L} & {len(g)} & {n_cr} & {f(raw, 3)} & {f(thr, 3)} & "
                    f"{f(p, 3)} & {f(r, 3)} \\\\")
    tex.append(r"""% --- Tabelle count-mechanism (6.1.2), ohne abgestuerzte Seeds ---------------
			\begin{table}[htbp]
				\centering
				\small
				\begin{tabular}{rrrllll}
					\toprule
					bag\_length & n & crashed & raw & thresholded	& patch\_level\_precision & patch\_level\_recall \\
					\midrule
""" + "\n".join(body) + r"""
					\bottomrule
				\end{tabular}
				\caption[Raw attention-weight \acs*{auc} vs. after thresholding with threshold $1/K$, median over seeds and bag-length from 150]{Raw attention-weight
				\ac{auc} vs. after thresholding with threshold $1/K$, median over the seeds that did not crash and bag-length from 150.
				\emph{n} counts the seeds with a patch-level \ac{auc}, \emph{crashed} the excluded seeds (Section~\ref{sec:eval}).}
				\label{tab:count-mechanism}
			\end{table}
""")

    # --- Tab. mnist-gated + 6.1.3 ---------------------------------------------
    l100 = df[(df["bag_length"] == 100) & (df["num_bags"].isin([400, 500]))]
    body = []
    print("\nTab. mnist-gated (Bags, Modell, n, crashed, patch-AUC [q25,q75], MAE [q25,q75]):")
    for i, nb in enumerate((400, 500)):
        if i:
            body.append(r"					\midrule")
        for m in ("attention", "gated_attention"):
            g = l100[(l100["num_bags"] == nb) & (l100["model"] == m)]
            go = g[g["crashed"] == 0]
            pa, mae = go["patch_level_auc"], go["count_mae"]
            print(f"   {nb} {m:16} n={len(go)} crashed={int(g['crashed'].sum())} "
                  f"pAUC {med(pa):.3f} [{pa.quantile(.25):.3f}, {pa.quantile(.75):.3f}] "
                  f"MAE {med(mae):.3f} [{mae.quantile(.25):.3f}, {mae.quantile(.75):.3f}]")
            body.append(f"					{nb} & {m.replace('_', chr(92) + '_')} & {len(go)} & "
                        f"{int(g['crashed'].sum())} & {f(med(pa), 3)} & {f(med(mae), 3)} \\\\")
        # Paarvergleich ueber die Seeds, die in beiden Varianten nicht abgestuerzt sind
        a = l100[(l100["num_bags"] == nb) & (l100["model"] == "attention")].set_index("run_name")
        b = l100[(l100["num_bags"] == nb) & (l100["model"] == "gated_attention")].set_index("run_name")
        shared = [s for s in SEED_ORDER if s in a.index and s in b.index
                  and not a.loc[s, "crashed"] and not b.loc[s, "crashed"]]
        for metric, better in (("patch_level_auc", np.greater), ("count_mae", np.less)):
            wins = sum(better(b.loc[s, metric], a.loc[s, metric]) for s in shared)
            diffs = [b.loc[s, metric] - a.loc[s, metric] for s in shared]
            print(f"     {nb} Bags, {metric}: gated besser in {wins}/{len(shared)} gemeinsamen Seeds "
                  f"({', '.join(shared)}), Differenzen gated-ungated "
                  + ", ".join(f"{d:+.3f}" for d in diffs))
    tex.append(r"""% --- Tabelle mnist-gated (6.1.3), ohne abgestuerzte Seeds -------------------
			\begin{table}[htbp]
				\centering
				\small
				\begin{tabular}{rlrrll}
					\toprule
					num\_bags & model & n & crashed & patch\_level\_auc & count\_mae \\
					\midrule
""" + "\n".join(body) + r"""
					\bottomrule
				\end{tabular}
				\caption[Counting performance of gated vs. ungated attention on \acs*{mnist} with bag size 100]{Counting performance of gated vs.
				ungated attention on \ac{mnist} with bag size 100, median over the seeds that did not crash (out of five).
				\emph{n} counts the seeds used, \emph{crashed} the excluded seeds (Section~\ref{sec:eval}).}
				\label{tab:mnist-gated}
			\end{table}
""")

    # --- Tab. mnist_activation: Dichtetest ------------------------------------
    dens = loading.load_runs(DB, ["Default"], level="child", status="all")
    dens = dens[dens["parent_run_name"] == "density_check"]
    print(f"\nTab. mnist_activation (density_check): {len(dens)} Seeds, "
          f"crashed {int(dens['crashed'].sum())}, min. Accuracy {dens['accuracy'].min():.3f}")


# --------------------------------------------------------------------------- #
# 3. Synthetische Drohnen-Bags (Abschnitt 6.5.2)                               #
# --------------------------------------------------------------------------- #
SYNTH_ROWS = [("attention_synth_bags_shuffled_rerun", r"\acs*{abmil} (softmax)"),
              ("gated_attention_synth_bags_shuffled_rerun", r"Gated \acs*{abmil}"),
              ("attention_batchnorm_synth_bags_shuffled_rerun", "Group normalisation"),
              ("attention_third_conv_synth_bags_shuffled_rerun", r"Third conv.\ layer"),
              ("attention_dropout_synth_bags_shuffled_rerun", "Dropout"),
              ("clam_synth_bags_shuffled_rerun", r"\acs*{clam}, $k=8$")]


def synth(tex):
    df = loading.load_runs(DB, ["synth_models"], level="child", status="all")
    print("\n== Synthetische Bags (Rerun) ==")
    body = []
    for parent, label in SYNTH_ROWS:
        g = df[df["parent_run_name"] == parent]
        go = g[g["crashed"] == 0]
        n_cr = int(g["crashed"].sum())
        vals = {k: med(go[k]) for k in ("accuracy", "auc", "patch_level_auc",
                                        "count_mae", "count_bias", "count_r2")}
        alls = {k: med(g[k]) for k in vals}
        print(f"   {parent:48} n={len(go)} crashed={n_cr} "
              + " ".join(f"{k}={alls[k]:.3f}->{vals[k]:.3f}" for k in vals))
        print("      je Seed MAE: " + ", ".join(
            f"{r.run_name}={r.count_mae:.3f}{'*' if r.crashed else ''}"
            for r in g.sort_values("run_name").itertuples()))
        if parent.startswith("clam"):
            # Die CLAM-Zeile zaehlt bei 0.5 bzw. kalibriert (nicht bei 1/K) und hat
            # keinen Crash -> Werte unveraendert aus der Thesis uebernehmen.
            body.append(r"					\acs*{clam}, $k=8$ & 1.000 & 1.000 & 0.996 & 1.45 & $-1.45$ & 0.994 & 0.66 & "
                        f"{n_cr}/{len(g)} \\\\")
            continue
        body.append(f"					{label} & {f(vals['accuracy'], 3)} & {f(vals['auc'], 3)} & "
                    f"{f(vals['patch_level_auc'], 3)} & {f(vals['count_mae'], 2)} & "
                    f"{signed(vals['count_bias'], 2)} & {f(vals['count_r2'], 3)} & --- & "
                    f"{n_cr}/{len(g)} \\\\")
    tex.append(r"""% --- Tabelle drone_synth (6.5.2), ohne abgestuerzte Seeds -------------------
			\begin{table}[htbp]
				\centering
				\footnotesize
				\setlength{\tabcolsep}{4pt}
				\begin{tabular}{@{}lcccccccc@{}}
					\toprule
					\multirow{2}{*}{Model} & \multirow{2}{*}{Accuracy} & \multirow{2}{*}{\acs*{auc}} & \multirow{2}{*}{patch \acs*{auc}} & \multicolumn{4}{c}{Count of positive patches} & \multirow{2}{*}{Crashed} \\
					\cmidrule(lr){5-8}
					& & & & \acs*{mae} & Bias & $R^2$ & \acs*{mae}$^{\text{cal}}$ & \\
					\midrule
""" + "\n".join(body) + r"""
					\bottomrule
				\end{tabular}
				\caption[All model variants on the synthetic drone bags, three seeds, median over the seeds that did not crash]{All model variants on the
				synthetic drone bags, three seeds, median over the seeds that did not crash.
				\acs*{mae}, bias and $R^2$ use the fixed threshold $1/K$ for the \ac{abmil} variants and $0.5$
				on the patch probabilities for \ac{clam}, \acs*{mae}$^{\text{cal}}$ uses the threshold calibrated on the validation split.
				\emph{Crashed} counts the excluded seeds that assign all test bags to the same class (Section~\ref{sec:eval}).
				Reference error of the always-zero predictor: $25.94$.}
				\label{tab:drone_synth}
			\end{table}
""")


# --------------------------------------------------------------------------- #
# 4. Drohne: Crashes vs. schwache Seeds (Tab. drone_bbg, drone_greedy)          #
# --------------------------------------------------------------------------- #
def drone():
    print("\n== Drohne ==")
    for exp in ("drone_test", "clam_greedy_drone", "clam_instscores", "pretrained_test",
                "clam_sweep_v3"):
        df = loading.load_runs(DB, [exp], level="child", status="all")
        print(f"   {exp:18} Seeds {len(df):>3}, crashed {int(df['crashed'].sum())}, "
              f"Acc <= 0.81: {int((df['accuracy'] <= 0.81).sum())}, "
              f"min. Acc {df['accuracy'].min():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    crash_audit(args.outdir)
    tex = ["% Erzeugt von eval/scripts/crash_exclusion.py -- Mediane ohne abgestuerzte Seeds.\n"
           "% Ersetzt die gleichnamigen Tabellen in masterarbeit.tex (Labels unveraendert).\n"]
    mnist(tex)
    synth(tex)
    drone()
    path = os.path.join(args.outdir, "tables_crash_excluded.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(tex))
    print("\ngeschrieben:", path)


if __name__ == "__main__":
    main()
