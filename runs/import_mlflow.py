#!/usr/bin/env python3
"""
import_mlflow.py

Importiert MLflow-Experiments und -Runs aus einem geclusterten mlruns-Verzeichnis
(nach rsync vom Cluster) in die lokale MLflow-Datenbank.

Folgendes wird vollständig übertragen:
  • Experiments (nach Name zusammengeführt, keine ID-Konflikte)
  • Parent- und Child-Runs (verschachtelte Struktur bleibt erhalten)
  • Metriken inkl. vollständiger Step-History (z.B. train_loss pro Epoche)
  • Parameter und Tags
  • Artifacts (Modelle, Plots, Tabellen, Config-Dateien)
  • Run-Status (FINISHED / FAILED) und Timestamps

Usage:
    python import_mlflow.py --source ./cluster_mlruns_staging --target ./mlruns
    python import_mlflow.py --source ./cluster_mlruns_staging  # target=./mlruns (default)
    python import_mlflow.py --source ./cluster_mlruns_staging --dry-run
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# Tags, die MLflow intern setzt und die wir nicht blind überschreiben dürfen
# (werden separat behandelt oder weggelassen)
SKIP_TAGS = {
    "mlflow.log-model.history",   # wird beim Artifact-Log automatisch gesetzt
}


def normalize_uri(path_or_uri: str) -> str:
    """Stellt sicher, dass der Tracking-URI ein gültiges Präfix trägt."""
    p = path_or_uri.strip()
    if p.startswith("sqlite:///") or p.startswith("http"):
        return p
    if not p.startswith("file://"):
        p = "file://" + str(Path(p).resolve())
    return p


def find_artifact_dir(
    src_run,
    source_artifacts_dir: str | None,
    src_mlruns_dir: str | None,
) -> Path:
    """
    Lokalen Artifact-Pfad für einen Run ermitteln.

    - SQLite-Modus:  source_artifacts_dir/<exp_id>/<run_id>/artifacts/
    - File-Modus:    src_mlruns_dir/<exp_id>/<run_id>/artifacts/
    """
    exp_id = src_run.info.experiment_id
    run_id = src_run.info.run_id
    if source_artifacts_dir:
        return Path(source_artifacts_dir) / exp_id / run_id / "artifacts/"
    if src_mlruns_dir:
        return Path(src_mlruns_dir) / exp_id / run_id / "artifacts/"
    return Path(src_run.info.artifact_uri.replace("file://", ""))


def import_run(
    src_client: MlflowClient,
    tgt_client: MlflowClient,
    src_run,
    tgt_experiment_id: str,
    run_id_map: dict,
    src_mlruns: str | None,
    source_artifacts_dir: str | None,
    dry_run: bool,
) -> str:
    """
    Importiert einen einzelnen Run vom Quell- in den Ziel-Client.
    Gibt die neue Run-ID im Ziel zurück.
    """
    src_id = src_run.info.run_id
    run_name = src_run.info.run_name or f"imported_{src_id[:8]}"

    # --- Tags vorbereiten ---
    tags = {}
    src_parent_id = None
    for key, value in src_run.data.tags.items():
        if key in SKIP_TAGS:
            continue
        if key == "mlflow.parentRunId":
            # Parent-Run-ID auf neue lokale ID remappen
            src_parent_id = value
            if value in run_id_map:
                tags[key] = run_id_map[value]
            # Falls Parent noch nicht importiert: wird unten nachgeholt
            continue
        tags[key] = value

    print(f"    → Run: '{run_name}' ({src_id[:8]}...)")

    if dry_run:
        print(f"      [dry-run] würde Run erstellen in Experiment {tgt_experiment_id}")
        return f"dry-run-{src_id}"

    # Neuen Run erstellen
    tgt_run = tgt_client.create_run(
        experiment_id=tgt_experiment_id,
        run_name=run_name,
        tags=tags,
    )
    tgt_id = tgt_run.info.run_id

    # --- Parent-ID nachtragen, falls der Parent jetzt bekannt ist ---
    if src_parent_id and src_parent_id in run_id_map:
        tgt_client.set_tag(tgt_id, "mlflow.parentRunId", run_id_map[src_parent_id])

    # --- Parameter ---
    params = src_run.data.params
    if params:
        # MLflow erlaubt max. 100 Params pro Batch-Aufruf
        param_list = list(params.items())
        for i in range(0, len(param_list), 100):
            batch = param_list[i:i + 100]
            for k, v in batch:
                tgt_client.log_param(tgt_id, k, v)
        print(f"      {len(params)} Parameter geloggt")

    # --- Metriken mit vollständiger Step-History ---
    metric_keys = list(src_run.data.metrics.keys())
    total_metric_points = 0
    for key in metric_keys:
        history = src_client.get_metric_history(src_id, key)
        for point in history:
            tgt_client.log_metric(
                tgt_id,
                key,
                point.value,
                timestamp=point.timestamp,
                step=point.step,
            )
            total_metric_points += 1
    if total_metric_points > 0:
        print(f"      {total_metric_points} Metrik-Punkte für {len(metric_keys)} Keys geloggt")

    # --- Artifacts ---
    src_artifact_dir = find_artifact_dir(src_run, source_artifacts_dir, src_mlruns)
    if src_artifact_dir.exists():
        tgt_artifact_uri = tgt_run.info.artifact_uri.replace("file://", "")
        tgt_artifact_dir = Path(tgt_artifact_uri)
        tgt_artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_count = 0
        for item in src_artifact_dir.iterdir():
            dest = tgt_artifact_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
            artifact_count += 1
        print(f"      {artifact_count} Artifact(s) kopiert")
    else:
        print(f"      Keine Artifacts gefunden unter {src_artifact_dir}")

    # --- Run abschließen ---
    status_map = {
        "FINISHED": mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FINISHED),
        "FAILED":   mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FAILED),
        "KILLED":   mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.KILLED),
    }
    src_status = src_run.info.status
    if src_status in status_map:
        tgt_client.set_terminated(
            tgt_id,
            status=src_status,
            end_time=src_run.info.end_time,
        )
    else:
        # Läuft noch oder unbekannt → als FINISHED markieren
        tgt_client.set_terminated(tgt_id, "FINISHED")

    return tgt_id


def import_experiment(
    src_client: MlflowClient,
    tgt_client: MlflowClient,
    src_exp,
    src_mlruns: str | None,
    source_artifacts_dir: str | None,
    dry_run: bool,
) -> int:
    """Importiert alle Runs eines Experiments."""

    print(f"\n  Experiment: '{src_exp.name}'")

    # Ziel-Experiment finden oder erstellen
    tgt_exp = tgt_client.get_experiment_by_name(src_exp.name)
    if tgt_exp is not None:
        tgt_exp_id = tgt_exp.experiment_id
        print(f"  Existiert lokal bereits (ID={tgt_exp_id}) – Runs werden hinzugefügt.")
    else:
        if dry_run:
            tgt_exp_id = "dry-run-exp-id"
            print(f"  [dry-run] Würde Experiment '{src_exp.name}' anlegen.")
        else:
            tgt_exp_id = tgt_client.create_experiment(src_exp.name)
            print(f"  Neu angelegt (ID={tgt_exp_id}).")

    # Alle Runs aus dem Quell-Experiment laden
    all_src_runs = src_client.search_runs(
        experiment_ids=[src_exp.experiment_id],
        run_view_type=mlflow.entities.ViewType.ALL,
    )

    if not all_src_runs:
        print("  Keine Runs gefunden.")
        return 0

    # Parent-Runs vor Child-Runs importieren, damit mlflow.parentRunId
    # korrekt remappt werden kann.
    parent_runs = [r for r in all_src_runs
                   if "mlflow.parentRunId" not in r.data.tags]
    child_runs  = [r for r in all_src_runs
                   if "mlflow.parentRunId" in r.data.tags]

    print(f"  {len(parent_runs)} Parent-Run(s), {len(child_runs)} Child-Run(s) gefunden.")

    run_id_map: dict = {}  # src_run_id → tgt_run_id

    for src_run in parent_runs + child_runs:
        tgt_id = import_run(
            src_client=src_client,
            tgt_client=tgt_client,
            src_run=src_run,
            tgt_experiment_id=tgt_exp_id,
            run_id_map=run_id_map,
            src_mlruns=src_mlruns,
            source_artifacts_dir=source_artifacts_dir,
            dry_run=dry_run,
        )
        run_id_map[src_run.info.run_id] = tgt_id

    return len(all_src_runs)


def main():
    parser = argparse.ArgumentParser(
        description="Importiert MLflow-Runs vom Cluster in die lokale MLflow-DB."
    )
    parser.add_argument(
        "--source",
        default="sqlite:///./cluster_mlruns_staging3/mlflow.db",
        help="Tracking-URI der Quelle. Beispiele:\n"
             "  sqlite:///./staging/mlflow.db   (SQLite nach rsync)\n"
             "  ./cluster_mlruns_staging        (altes file-Backend)",
    )
    parser.add_argument(
        "--source-artifacts",
        default="./staging/mlruns",
        metavar="DIR",
        help="Lokaler Pfad zum gersynten mlartifacts/- oder mlruns/-Verzeichnis "
             "(nur bei SQLite-Quelle nötig, z.B. ./staging/mlartifacts).",
    )
    parser.add_argument(
        "--target",
        default="sqlite:///./mlflow.db",
        help="Lokale MLflow Tracking URI / mlruns-Verzeichnis (default: ./mlruns).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Nur dieses Experiment importieren (nach Name). "
             "Ohne Angabe werden alle Experiments importiert.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simuliert den Import ohne tatsächlich zu schreiben.",
    )
    args = parser.parse_args()

    src_uri = normalize_uri(args.source)
    tgt_uri = normalize_uri(args.target)

    # Für file-basierte Quellen: Pfad für Artifact-Rekonstruktion
    src_mlruns = None
    if src_uri.startswith("file://"):
        src_mlruns = src_uri.replace("file://", "")
        if not Path(src_mlruns).exists():
            print(f"FEHLER: Quell-Verzeichnis nicht gefunden: {src_mlruns}", file=sys.stderr)
            sys.exit(1)

    source_artifacts_dir = args.source_artifacts

    print("=" * 60)
    print("MLflow Import: Cluster → Lokal")
    print(f"  Quelle:  {src_uri}")
    if source_artifacts_dir:
        print(f"  Artifacts: {source_artifacts_dir}")
    print(f"  Ziel:    {tgt_uri}")
    if args.dry_run:
        print("  Modus:   DRY-RUN (kein Schreiben)")
    print("=" * 60)

    src_client = MlflowClient(tracking_uri=src_uri)
    tgt_client = MlflowClient(tracking_uri=tgt_uri)

    # Experiments aus Quelle laden
    all_experiments = src_client.search_experiments()

    # "Default"-Experiment überspringen (ID=0, enthält meist nichts Relevantes)
    experiments_to_import = [
        e for e in all_experiments
        if e.name != "Default" and (args.experiment is None or e.name == args.experiment)
    ]

    if not experiments_to_import:
        print("Keine passenden Experiments in der Quelle gefunden.")
        sys.exit(0)

    total_runs = 0
    for exp in experiments_to_import:
        n = import_experiment(
            src_client=src_client,
            tgt_client=tgt_client,
            src_exp=exp,
            src_mlruns=src_mlruns,
            source_artifacts_dir=source_artifacts_dir,
            dry_run=args.dry_run,
        )
        total_runs += n

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"[DRY-RUN] Würde {total_runs} Run(s) aus "
              f"{len(experiments_to_import)} Experiment(s) importieren.")
    else:
        print(f"Fertig! {total_runs} Run(s) aus "
              f"{len(experiments_to_import)} Experiment(s) importiert.")
    print(f"MLflow UI starten:  mlflow ui --backend-store-uri {tgt_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()