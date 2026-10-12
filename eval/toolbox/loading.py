"""Laden, Flachklopfen und Filtern von MLflow-Runs.

Alle Sweeps in diesem Projekt loggen zweistufig: ein Parent-Run je Konfiguration
mit den ueber die Seeds aggregierten Metriken (``<metrik>_mean`` / ``_std``) und
darunter ein Child-Run je Seed mit den Rohwerten.  Modell- und Dataset-Parameter
stehen dabei nur am Parent -- ``level="child"`` reicht sie deshalb standardmaessig
an die Children herunter (``inherit_params``).
"""

from __future__ import annotations

import os
import re
import warnings

import mlflow
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

PARENT_COL = "tags.mlflow.parentRunId"
NAME_COL = "tags.mlflow.runName"

#: Spalten, die beim Flachklopfen keinen kurzen Alias bekommen (reines Rauschen).
_BORING_TAGS = {
    "mlflow.source.name", "mlflow.source.git.commit", "mlflow.source.type",
    "mlflow.user", "mlflow.loggedArtifacts", "mlflow.parentRunId",
}


# --------------------------------------------------------------------------- #
# Verbindung / Discovery                                                       #
# --------------------------------------------------------------------------- #
def tracking_uri(db: str) -> str:
    """Akzeptiert einen Pfad zur SQLite-Datei oder eine fertige Tracking-URI."""
    if "://" in db:
        return db
    return "sqlite:///" + os.path.abspath(db)


def connect(db: str) -> None:
    mlflow.set_tracking_uri(tracking_uri(db))


def list_experiments(db: str) -> pd.DataFrame:
    """Alle Experimente mit Run-Zahl, sortiert nach ID."""
    connect(db)
    client = mlflow.MlflowClient()
    rows = []
    for exp in client.search_experiments():
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                                  output_format="pandas")
        n_parent = int(runs[PARENT_COL].isna().sum()) if PARENT_COL in runs else len(runs)
        rows.append({
            "experiment_id": int(exp.experiment_id),
            "name": exp.name,
            "runs": len(runs),
            "parents": n_parent,
            "children": len(runs) - n_parent,
        })
    return pd.DataFrame(rows).sort_values("experiment_id").reset_index(drop=True)


def resolve_experiments(db: str, wanted) -> list:
    """Experimentnamen oder -IDs -> Liste von Experiment-IDs."""
    connect(db)
    client = mlflow.MlflowClient()
    known = {e.name: e.experiment_id for e in client.search_experiments()}
    known_ids = {str(v) for v in known.values()}
    out = []
    for w in wanted:
        w = str(w)
        if w in known:
            out.append(known[w])
        elif w in known_ids:
            out.append(w)
        else:
            raise SystemExit(
                f"Experiment '{w}' nicht gefunden. Verfuegbar:\n  "
                + "\n  ".join(sorted(known)))
    return out


# --------------------------------------------------------------------------- #
# Flachklopfen                                                                 #
# --------------------------------------------------------------------------- #
def _numeric(series: pd.Series) -> pd.Series:
    """Wandelt in Zahlen um, wenn das fuer die Mehrheit der Werte gelingt."""
    conv = pd.to_numeric(series, errors="coerce")
    present = series.notna().sum()
    if present and conv.notna().sum() >= 0.8 * present:
        return conv
    return series


def flatten(df: pd.DataFrame) -> tuple:
    """Legt neben ``params.x``/``metrics.x``/``tags.x`` kurze Aliase an.

    Rueckgabe: (DataFrame, Alias-Herkunftstabelle).  Bei Namenskollision gewinnt
    ``params`` vor ``metrics`` vor ``tags``; die lange Schreibweise bleibt immer
    zusaetzlich nutzbar, sodass ``--filter "params.model=..."`` nie mehrdeutig ist.
    """
    origin, columns = {}, {}

    # Lange Schreibweise numerisch machen, wo sinnvoll.
    for col in df.columns:
        if col.startswith(("params.", "metrics.")):
            columns[col] = _numeric(df[col])

    # Echte Run-Spalten von MLflow (status, run_id, start_time, ...) duerfen
    # niemals von einem Alias verdeckt werden -- sonst filtert z.B.
    # --filter "status=..." unbemerkt auf tags.status statt auf den Run-Status.
    reserved = {c for c in df.columns if not c.startswith(("params.", "metrics.", "tags."))}

    # Kurze Aliase, Praezedenz params > metrics > tags.
    for prefix in ("params.", "metrics.", "tags."):
        for col in df.columns:
            if not col.startswith(prefix):
                continue
            short = col[len(prefix):]
            if short in origin or short in reserved:
                continue
            if prefix == "tags." and short in _BORING_TAGS:
                continue
            origin[short] = col
            columns[short] = columns.get(col, df[col])

    out = pd.concat([df.drop(columns=[c for c in columns if c in df.columns]),
                     pd.DataFrame(columns, index=df.index)], axis=1)
    meta = pd.DataFrame({"alias": list(origin), "quelle": list(origin.values())})
    return out, meta.sort_values("alias").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Laden                                                                        #
# --------------------------------------------------------------------------- #
def load_runs(db: str, experiments, level: str = "parent",
              inherit_params: bool = True, status: str = "FINISHED") -> pd.DataFrame:
    """Laedt Runs eines oder mehrerer Experimente als flache Tabelle.

    level    : ``parent`` (aggregierte Konfigurationen), ``child`` (Runs je Seed)
               oder ``all``.
    status   : ``FINISHED``, ``FAILED``, ... oder ``all``.
    """
    exp_ids = resolve_experiments(db, experiments)
    df = mlflow.search_runs(experiment_ids=exp_ids, output_format="pandas")
    if not len(df):
        raise SystemExit("Keine Runs gefunden.")

    if PARENT_COL not in df.columns:
        df[PARENT_COL] = pd.NA
    is_child = df[PARENT_COL].notna()

    if level == "child" and inherit_params:
        parents = df[~is_child].set_index("run_id")
        inherit = [c for c in df.columns
                   if c.startswith("params.") or c in (NAME_COL, "tags.dataset",
                                                       "tags.model")]
        for col in inherit:
            if col not in parents.columns:
                continue
            filled = df[PARENT_COL].map(parents[col])
            if col == NAME_COL:                       # Parent-Name separat halten
                df["parent_run_name"] = filled
            else:
                df[col] = df[col].where(df[col].notna(), filled)

    if level == "parent":
        df = df[~is_child]
    elif level == "child":
        df = df[is_child]
    elif level != "all":
        raise SystemExit(f"Unbekanntes --level '{level}' (parent|child|all)")

    if status != "all":
        df = df[df["status"].str.upper() == status.upper()]

    if not len(df):
        raise SystemExit(f"Keine Runs mit level={level}, status={status}.")

    out, _ = flatten(df.reset_index(drop=True))
    out["run_name"] = out.get(NAME_COL)
    return mark_crashed(out)


# --------------------------------------------------------------------------- #
# Abgestuerzte Seeds                                                           #
# --------------------------------------------------------------------------- #
def mark_crashed(df: pd.DataFrame) -> pd.DataFrame:
    """Spalte ``crashed`` (0/1): der Seed ordnet alle Test-Bags derselben Klasse zu.

    Alle negativ: Precision = Recall = 0 und kein Bag ueberzaehlt (Bias = -MAE).
    Precision = Recall = 0 allein reicht nicht -- sie entstehen auch, wenn nur
    negative Bags faelschlich positiv sind, und ein solcher Bag bekommt beim
    Zaehlen mindestens einen Patch.  Ohne Zaehlmetriken entscheidet P = R = 0.
    Alle positiv: Recall = 1 und Accuracy = Precision (beide = Anteil positiver
    Bags); Precision = 1 schliesst den fehlerfreien Lauf aus.
    Parent-Runs und Runs ohne Precision/Recall bleiben 0.
    """
    def col(name):
        return df[name] if name in df.columns else pd.Series(float("nan"), index=df.index)

    acc, prec, rec = col("accuracy"), col("precision"), col("recall")
    mae, bias = col("count_mae"), col("count_bias")
    no_overcount = (mae + bias).abs().lt(1e-6) | mae.isna() | bias.isna()
    all_negative = prec.eq(0) & rec.eq(0) & no_overcount
    all_positive = rec.eq(1) & prec.lt(1) & (acc - prec).abs().lt(1e-9)
    is_child = df[PARENT_COL].notna() if PARENT_COL in df.columns else True
    df["crashed"] = ((all_negative | all_positive) & is_child).astype(int)
    return df


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Welche Spalten gibt es, welchen Typ und wie viele Werte haben sie?"""
    rows = []
    for col in df.columns:
        if col.startswith(("params.", "metrics.", "tags.")):
            continue
        s = df[col]
        n = int(s.notna().sum())
        if not n:
            continue
        kind = ("metrik" if "metrics." + col in df.columns else
                "param" if "params." + col in df.columns else
                "tag" if "tags." + col in df.columns else "meta")
        uniq = s.dropna().unique()
        example = ", ".join(str(v) for v in uniq[:4])
        if len(uniq) > 4:
            example += ", ..."
        rows.append({"spalte": col, "art": kind, "werte": n,
                     "unique": len(uniq), "numerisch": pd.api.types.is_numeric_dtype(s),
                     "beispiele": example[:70]})
    return pd.DataFrame(rows).sort_values(["art", "spalte"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Abgeleitete Spalten                                                          #
# --------------------------------------------------------------------------- #
def derive(df: pd.DataFrame, specs) -> pd.DataFrame:
    """Zieht neue Spalten per Regex aus einer bestehenden Spalte.

    Spec-Syntax: ``ziel1[,ziel2]=quellspalte:regex`` -- jede Capture-Group des
    Regex fuellt eine Zielspalte, z.B.::

        num_bags,bag_length=dataset:bags(\\d+)_len(\\d+)
    """
    for spec in specs or []:
        try:
            targets, rest = spec.split("=", 1)
            source, pattern = rest.split(":", 1)
        except ValueError:
            raise SystemExit(
                f"--derive '{spec}' ungueltig. Erwartet: ziel[,ziel]=quelle:regex")
        targets = [t.strip() for t in targets.split(",")]
        if source not in df.columns:
            raise SystemExit(f"--derive: Quellspalte '{source}' gibt es nicht.")
        extracted = df[source].astype(str).str.extract(re.compile(pattern))
        if extracted.shape[1] != len(targets):
            raise SystemExit(
                f"--derive '{spec}': {extracted.shape[1]} Capture-Groups, "
                f"aber {len(targets)} Zielspalten.")
        for i, target in enumerate(targets):
            df[target] = _numeric(extracted.iloc[:, i])
    return df


# --------------------------------------------------------------------------- #
# Filter                                                                       #
# --------------------------------------------------------------------------- #
_OPS = ["!~", ">=", "<=", "!=", "~", "=", ">", "<"]


def _parse_filter(expr: str):
    for op in _OPS:
        idx = expr.find(op)
        if idx > 0:
            return expr[:idx].strip(), op, expr[idx + len(op):].strip()
    raise SystemExit(
        f"--filter '{expr}' ungueltig. Erwartet: spalte OP wert "
        f"mit OP aus = != > >= < <= ~ !~ (~ = Regex, Werteliste mit Komma bei =/!=)")


def apply_filters(df: pd.DataFrame, filters, query: str = None) -> pd.DataFrame:
    """Wendet ``--filter``-Ausdruecke und optional eine pandas-``--query`` an."""
    for expr in filters or []:
        col, op, val = _parse_filter(expr)
        if col not in df.columns:
            raise SystemExit(
                f"--filter '{expr}': Spalte '{col}' gibt es nicht. "
                f"'columns' zeigt die verfuegbaren Spalten.")
        s = df[col]
        if op in ("~", "!~"):
            hit = s.astype(str).str.contains(val, regex=True, na=False)
            df = df[hit if op == "~" else ~hit]
            continue
        if op in ("=", "!=") and "," in val:
            wanted = [v.strip() for v in val.split(",")]
            if pd.api.types.is_numeric_dtype(s):
                wanted = pd.to_numeric(pd.Series(wanted), errors="coerce").tolist()
            hit = s.isin(wanted)
            df = df[hit if op == "=" else ~hit]
            continue
        cast = pd.to_numeric(val, errors="coerce") if pd.api.types.is_numeric_dtype(s) else val
        if pd.isna(cast) if not isinstance(cast, str) else False:
            raise SystemExit(f"--filter '{expr}': '{val}' ist keine Zahl, "
                             f"Spalte '{col}' aber numerisch.")
        ops = {"=": s.__eq__, "!=": s.__ne__, ">": s.__gt__,
               ">=": s.__ge__, "<": s.__lt__, "<=": s.__le__}
        df = df[ops[op](cast)]

    if query:
        df = df.query(query)

    if not len(df):
        raise SystemExit("Filter lassen keine Runs uebrig.")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrik-Historie (Lernkurven)                                                 #
# --------------------------------------------------------------------------- #
def load_history(db: str, df: pd.DataFrame, keys, carry=None) -> pd.DataFrame:
    """Schrittweise Historie der genannten Metriken fuer alle Runs in ``df``.

    ``search_runs`` liefert nur den zuletzt geloggten Wert je Metrik; fuer
    Lernkurven muss die Historie einzeln nachgeladen werden.  ``carry`` nennt die
    Spalten aus ``df``, die mitgefuehrt werden sollen (z.B. die Serien- oder
    Facettenspalte) -- sie werden erst nach dem Pivot wieder angeheftet, damit
    fehlende Werte darin keine Messpunkte verschlucken.
    """
    connect(db)
    client = mlflow.MlflowClient()

    rows = []
    for run_id in df["run_id"]:
        for key in keys:
            for point in client.get_metric_history(run_id, key):
                rows.append((run_id, key, point.step, point.value))
    if not rows:
        raise SystemExit(f"Keine Metrik-Historie fuer {', '.join(keys)} gefunden.")

    long = pd.DataFrame(rows, columns=["run_id", "metrik", "step", "value"])
    wide = long.pivot_table(index=["run_id", "step"], columns="metrik",
                            values="value").reset_index()
    wide.columns.name = None

    keep = ["run_id"] + [c for c in dict.fromkeys(
        [c for c in ("run_name", "parent_run_name") if c in df.columns]
        + [c for c in (carry or []) if c in df.columns]) if c != "run_id"]
    return wide.merge(df[keep].drop_duplicates("run_id"), on="run_id", how="left")
