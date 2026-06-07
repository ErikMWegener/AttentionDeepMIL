#!/bin/bash
# sync_mlflow.sh
#
# Überträgt MLflow-Daten vom Cluster auf den lokalen Rechner.
# Anschließend import_mlflow.py ausführen, um sie in die lokale DB zu integrieren.
#
# Usage:  bash sync_mlflow.sh [JOB_ID]
#         JOB_ID ist optional – dient nur zur Protokollierung.

set -euo pipefail

# ============================================================
# Konfiguration – hier anpassen
# ============================================================
CLUSTER_USER="<5-Steller>"
CLUSTER_HOST="mcgarret.informatik.uni-halle.de"
PROJECT_NAME="mil_project"
CLUSTER_MLRUNS="/zpool1/slurm_data/${CLUSTER_USER}/${PROJECT_NAME}/mlruns"

# Lokales Zielverzeichnis (Staging-Bereich, NICHT direkt mlruns!)
LOCAL_STAGING="./cluster_mlruns_staging"
# ============================================================

JOB_ID="${1:-unbekannt}"
echo "=== MLflow Sync vom Cluster ==="
echo "    Cluster:  ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_MLRUNS}"
echo "    Lokal:    ${LOCAL_STAGING}"
echo "    Job-ID:   ${JOB_ID}"
echo ""

# Staging-Verzeichnis anlegen
mkdir -p "${LOCAL_STAGING}"

# rsync:
#   -a  → Archiv-Modus (Rechte, Timestamps, rekursiv erhalten)
#   -v  → verbose
#   -z  → Komprimierung während der Übertragung
#   --progress → Fortschritt anzeigen
#   --delete   → auf Staging-Seite löschen was auf Cluster nicht mehr existiert
rsync -avz --progress --delete \
  "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_MLRUNS}/" \
  "${LOCAL_STAGING}/"

echo ""
echo "=== Sync abgeschlossen: $(date) ==="
echo ""
echo "Nächster Schritt – Runs in lokale MLflow-DB importieren:"
echo ""
echo "  python import_mlflow.py \\"
echo "    --source \"${LOCAL_STAGING}\" \\"
echo "    --target \"./mlruns\""
echo ""
echo "Oder mit expliziter file://-URI:"
echo "  python import_mlflow.py \\"
echo "    --source \"file://\$(pwd)/${LOCAL_STAGING}\" \\"
echo "    --target \"file://\$(pwd)/mlruns\""
