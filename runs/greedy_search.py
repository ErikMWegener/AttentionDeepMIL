#!/usr/bin/env python3
"""
greedy_search.py
Greedy Coordinate Descent für die Hyperparameter-Optimierung des AttentionDeepMIL-Modells.

Optimiert koordinatenweise: model_num_maps, model_kernel_size, model_pool_size,
model_M, model_L. Pro Parameter werden alle Kandidatenwerte getestet, der beste
wird fixiert, danach der nächste Parameter optimiert.

Ablauf pro Trial:
  1. main_mlflow.py wird als Subprozess gestartet (--config lädt die Basis-Config,
     die fünf --model_* Argumente überschreiben deren Defaults).
  2. Die aggregierte Zielmetrik wird aus dem MLflow Parent-Run gelesen.

Nutzt MLFLOW_TRACKING_URI aus der Umgebung (im Slurm-Skript gesetzt). Der Treiber
und alle Subprozesse schreiben/lesen damit dieselbe SQLite-DB.
"""

import argparse
import sys
import subprocess
import time

import mlflow
from mlflow.tracking import MlflowClient


# ── Standard-Suchräume ───────────────────────────────────────────────────────
DEFAULT_SEARCH_SPACE = {
    "model_num_maps":    [64, 128, 256],
    #"model_kernel_size": [5, 7, 9],      # nur ungerade (padding = k//2 bleibt symmetrisch)
    "model_pool_size":   [4, 6, 8],
    "model_M":           [250, 500, 1024, 2048],  # 1048 = sqrt(1M), 2048 = nächster Zweierpotenz
    #"model_L":           [64, 128, 256],
}

# Reihenfolge: Feature-Extraktor zuerst (bestimmt die Repräsentation),
# dann die Dimensionen des Attention-Mechanismus.
DEFAULT_ORDER = ["model_num_maps", "model_pool_size", "model_M"]

DEFAULT_BASELINE = {
    "model_num_maps":    50,
    "model_kernel_size": 5,
    "model_pool_size":   4,
    "model_M":           500,
    "model_L":           128,
}


def parse_target_metric(spec):
    """
    Parst die Zielmetrik-Spezifikation in eine Liste (name, gewicht).

    "auc_mean"
        -> [("auc_mean", 1.0)]
    "auc_mean:0.5,counting_accuracy_mean:0.5"
        -> [("auc_mean", 0.5), ("counting_accuracy_mean", 0.5)]
    Negative Gewichte für zu minimierende Metriken:
    "auc_mean:1.0,counting_mae_mean:-0.2"
        -> MAE wird vom Score abgezogen.
    """
    targets = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, w = part.rsplit(":", 1)
            targets.append((name.strip(), float(w)))
        else:
            targets.append((part, 1.0))
    return targets


def run_trial(params, args, run_name):
    """Startet main_mlflow.py als Subprozess mit den gegebenen Hyperparametern."""
    cmd = [
        sys.executable, "main_mlflow.py",
        "--config", args.config,
        "--exp_name", args.exp_name,
        "--run_name", run_name,
        "--attention_activation", args.attention_activation,
        "--seeds", *[str(s) for s in args.seeds],
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.naive_counting:
        cmd.append("--naive_counting")
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    print(f"\n  -> Trial: {run_name}")
    print(f"     Params: {params}")
    t0 = time.time()
    result = subprocess.run(cmd)
    print(f"     fertig (exit={result.returncode}, {time.time()-t0:.0f}s)")
    return result.returncode


def read_score(client, experiment_id, run_name, targets):
    """
    Liest die gewichtete Zielmetrik aus dem Parent-Run mit dem gegebenen Namen.
    Robustes Python-seitiges Matching statt fragilem Filter-String.
    Gibt None zurück, wenn der Run oder eine Metrik fehlt.
    """
    runs = client.search_runs(
        [experiment_id],
        run_view_type=mlflow.entities.ViewType.ALL,
        order_by=["attribute.start_time DESC"],
        max_results=2000,
    )
    target_run = next(
        (r for r in runs if r.data.tags.get("mlflow.runName") == run_name),
        None,
    )
    if target_run is None:
        print(f"     WARN: Kein Run mit Namen '{run_name}' gefunden.")
        return None

    metrics = target_run.data.metrics
    score = 0.0
    for name, w in targets:
        if name not in metrics:
            print(f"     WARN: Metrik '{name}' fehlt. Verfügbar: {sorted(metrics.keys())}")
            return None
        score += w * metrics[name]
    return score


def main():
    parser = argparse.ArgumentParser(description="Greedy Coordinate Descent für MIL-Hyperparameter")
    parser.add_argument("--config", required=True, help="Pfad zur YAML-Basis-Config")
    parser.add_argument("--exp_name", default="greedy_search", help="MLflow Experiment-Name")
    parser.add_argument("--attention_activation", default="softmax")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1],
                        help="Seeds während der Suche (wenige = schneller)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochen pro Trial (überschreibt Config; weniger = schneller)")
    parser.add_argument("--naive_counting", action="store_true", default=False,
                        help="nötig, falls counting_*/patch_level_auc als Zielmetrik dient")
    parser.add_argument("--target_metric", default="auc_mean",
                        help="Zielmetrik(en), z.B. 'auc_mean' oder "
                             "'auc_mean:0.5,counting_accuracy_mean:0.5'")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Anzahl Durchläufe über alle Parameter (default 1)")
    parser.add_argument("--run_prefix", default="greedy")
    parser.add_argument("--final_seeds", nargs="+", type=int, default=None,
                        help="Seeds für einen finalen Lauf der besten Config (optional)")
    parser.add_argument("--params", nargs="+", default=None,
                        choices=list(DEFAULT_ORDER),
                        help="Welche Parameter optimiert werden (default: alle). "
                             "Nicht gewählte Achsen werden NICHT übergeben und "
                             "stammen damit aus der --config-Datei.")
    args = parser.parse_args()

    targets = parse_target_metric(args.target_metric)
    # Zu optimierende Achsen in fester Reihenfolge; der Rest bleibt fix aus der Config.
    selected = args.params if args.params else list(DEFAULT_ORDER)
    order = [p for p in DEFAULT_ORDER if p in selected]
    print(f"Zielmetrik: {targets}")
    print(f"Optimiere: {order}")

    client = MlflowClient()  # nutzt MLFLOW_TRACKING_URI
    mlflow.set_experiment(args.exp_name)  # legt Experiment an, falls nicht vorhanden
    experiment_id = client.get_experiment_by_name(args.exp_name).experiment_id

    # Startpunkt: nur die gewählten Achsen. Fixierte Parameter tauchen nicht in
    # best_config auf und werden daher nicht als --model_* übergeben.
    best_config = {p: DEFAULT_BASELINE[p] for p in order}
    cache = {}          # tuple(sorted(config.items())) -> score
    history = []
    trial_idx = [0]     # Liste, um in der Closure mutierbar zu sein

    def evaluate(config):
        key = tuple(sorted(config.items()))
        if key in cache:
            print(f"     (gecacht: score={cache[key]:.4f})")
            return cache[key]
        run_name = f"{args.run_prefix}_t{trial_idx[0]:03d}_" + "_".join(
            f"{k.replace('model_', '')}{v}" for k, v in sorted(config.items()))
        run_trial(config, args, run_name)
        score = read_score(client, experiment_id, run_name, targets)
        if score is None:
            score = float("-inf")  # fehlgeschlagene/leere Runs sind nie "best"
        cache[key] = score
        history.append((dict(config), score))
        trial_idx[0] += 1
        return score

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\n=== Baseline ===")
    best_score = evaluate(best_config)
    print(f"Baseline-Score: {best_score:.4f}")

    # ── Greedy-Runden ────────────────────────────────────────────────────────
    for rnd in range(args.rounds):
        print(f"\n{'#'*52}\n# Runde {rnd+1}/{args.rounds}\n{'#'*52}")
        improved = False

        for param in order:
            print(f"\n--- {param} (aktuell={best_config[param]}) ---")
            param_best_value = best_config[param]
            param_best_score = best_score

            for value in DEFAULT_SEARCH_SPACE[param]:
                if value == best_config[param]:
                    continue
                trial_config = dict(best_config)
                trial_config[param] = value
                score = evaluate(trial_config)
                print(f"     {param}={value}: score={score:.4f}")
                if score > param_best_score:
                    param_best_score = score
                    param_best_value = value

            if param_best_value != best_config[param]:
                print(f"  [+] {param}: {best_config[param]} -> {param_best_value} "
                      f"(score {best_score:.4f} -> {param_best_score:.4f})")
                best_config[param] = param_best_value
                best_score = param_best_score
                improved = True
            else:
                print(f"  [=] {param}: bleibt bei {best_config[param]}")

        if not improved:
            print(f"\nKeine Verbesserung in Runde {rnd+1} -- Konvergenz.")
            break

    # ── Ergebnis ─────────────────────────────────────────────────────────────
    print(f"\n{'='*52}\nBESTE KONFIGURATION\n{'='*52}")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"  Score ({args.target_metric}): {best_score:.4f}")

    print("\nTop-5 Konfigurationen:")
    for cfg, sc in sorted(history, key=lambda x: x[1], reverse=True)[:5]:
        short = {k.replace("model_", ""): v for k, v in cfg.items()}
        print(f"  score={sc:.4f}  {short}")

    # ── Optionaler finaler Lauf mit vollen Seeds ─────────────────────────────
    if args.final_seeds:
        print(f"\n=== Finaler Lauf mit Seeds {args.final_seeds} ===")
        final_args = argparse.Namespace(**vars(args))
        final_args.seeds = args.final_seeds
        run_name = f"{args.run_prefix}_FINAL_" + "_".join(
            f"{k.replace('model_', '')}{v}" for k, v in sorted(best_config.items()))
        run_trial(best_config, final_args, run_name)
        final_score = read_score(client, experiment_id, run_name, targets)
        print(f"Finaler Score: {final_score}")


if __name__ == "__main__":
    main()