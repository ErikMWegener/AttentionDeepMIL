"""Aggregation und Export beliebiger Ergebnistabellen."""

from __future__ import annotations

import pandas as pd

#: Aggregatnamen, die ``--agg`` versteht.
AGGS = ("mean", "std", "median", "min", "max", "count", "sem", "sum",
        "q25", "q75", "iqr", "mad")

#: Aggregate, die pandas nicht als Name kennt.
_FUNCS = {
    "q25": lambda s: s.quantile(0.25),
    "q75": lambda s: s.quantile(0.75),
    "iqr": lambda s: s.quantile(0.75) - s.quantile(0.25),
    "mad": lambda s: (s - s.median()).abs().median(),   # robuste Streuung
}


def aggregate(df: pd.DataFrame, group_by, metrics, aggs=("mean", "std"),
              exclude_crashed: bool = False) -> pd.DataFrame:
    """Gruppiert und aggregiert; ohne ``group_by`` bleiben die Runs einzeln stehen.

    Ergebnisspalten heissen ``<metrik>_<agg>``; zusaetzlich zaehlt ``n`` die Runs
    je Gruppe, denn eine Standardabweichung ohne Fallzahl ist nicht lesbar.
    ``exclude_crashed`` aggregiert nur ueber die nicht abgestuerzten Seeds
    (Spalte ``crashed``, siehe ``loading.mark_crashed``); ``n`` zaehlt dann nur
    diese, die Spalte ``crashed`` die herausgenommenen.
    """
    full = df
    if exclude_crashed and "crashed" in df.columns:
        df = df[df["crashed"] == 0]
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise SystemExit(f"Unbekannte Metriken: {', '.join(missing)}. "
                         f"'columns' zeigt die verfuegbaren Spalten.")
    for agg in aggs:
        if agg not in AGGS:
            raise SystemExit(f"Unbekanntes --agg '{agg}' (erlaubt: {', '.join(AGGS)})")

    if not group_by:
        keep = [c for c in ("run_name", "run_id") if c in df.columns]
        return df[keep + list(metrics)].reset_index(drop=True)

    missing = [g for g in group_by if g not in df.columns]
    if missing:
        raise SystemExit(f"Unbekannte --group-by Spalten: {', '.join(missing)}")

    grouped = df.groupby(list(group_by), dropna=False)
    out = grouped.size().rename("n").reset_index()
    for metric in metrics:
        stats = grouped[metric].agg([_FUNCS.get(a, a) for a in aggs])
        if isinstance(stats, pd.Series):
            stats = stats.to_frame(aggs[0])
        stats.columns = [f"{metric}_{a}" for a in aggs]
        # Wie viele Runs haben fuer diese Metrik ueberhaupt einen Wert?
        stats[f"{metric}_n"] = grouped[metric].count().values
        out = out.merge(stats.reset_index(), on=list(group_by), how="left")

    # Die Fallzahl je Metrik nur zeigen, wo sie von der Gruppengroesse abweicht
    # (fehlende Werte, z.B. nicht berechenbare Patch-AUC) -- sonst ist sie Rauschen.
    for metric in metrics:
        col = f"{metric}_n"
        if (out[col] == out["n"]).all():
            out = out.drop(columns=col)

    # Gruppen, in denen jeder Seed abgestuerzt ist, bleiben mit n = 0 stehen.
    if full is not df:
        crashed = (full.groupby(list(group_by), dropna=False)["crashed"].sum()
                   .rename("crashed").reset_index())
        out = crashed.merge(out, on=list(group_by), how="left")
        out["n"] = out["n"].fillna(0).astype(int)
        rest = [c for c in out.columns if c not in (*group_by, "n", "crashed")]
        out = out[list(group_by) + ["n", "crashed"] + rest]
    return out.sort_values(list(group_by)).reset_index(drop=True)


def _fmt(value, precision):
    if pd.isna(value):
        return "--"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def render(df: pd.DataFrame, precision: int = 3, pm: bool = True) -> pd.DataFrame:
    """Formatiert zur Anzeige und fasst ``x_mean``/``x_std`` zu ``mean ± std``."""
    out = pd.DataFrame(index=df.index)
    consumed = set()
    for col in df.columns:
        if col in consumed:
            continue
        if pm and col.endswith(("_mean", "_median")):
            base = col.rsplit("_", 1)[0]
            q25, q75 = base + "_q25", base + "_q75"
            spread = base + ("_std" if col.endswith("_mean") else "_mad")

            # Median mit Quartilen: "median [q25, q75]" -- der IQR ist
            # asymmetrisch und liesse sich als ± nicht ehrlich schreiben.
            if col.endswith("_median") and q25 in df.columns and q75 in df.columns:
                consumed |= {q25, q75}
                out[base] = [
                    "--" if pd.isna(m) else
                    f"{m:.{precision}f}" if pd.isna(a) or pd.isna(b) else
                    f"{m:.{precision}f} [{a:.{precision}f}, {b:.{precision}f}]"
                    for m, a, b in zip(df[col], df[q25], df[q75])]
                continue

            if spread in df.columns:
                consumed.add(spread)
                out[base] = [
                    "--" if pd.isna(m) else
                    f"{m:.{precision}f}" if pd.isna(s) else
                    f"{m:.{precision}f} ± {s:.{precision}f}"
                    for m, s in zip(df[col], df[spread])]
                continue
        out[col] = [_fmt(v, precision) for v in df[col]]
    return out


def to_text(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def to_markdown(df: pd.DataFrame) -> str:
    """Markdown ohne die optionale ``tabulate``-Abhaengigkeit."""
    cols = [str(c) for c in df.columns]
    widths = [max(len(c), *(len(str(v)) for v in df[c])) if len(df) else len(c)
              for c in df.columns]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]).ljust(w)
                                       for c, w in zip(df.columns, widths)) + " |")
    return "\n".join(lines) + "\n"


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "",
             group_col: str = None) -> str:
    """booktabs-Tabelle; ``group_col`` zieht eine Linie bei jedem Wertwechsel."""
    def esc(text):
        text = str(text)
        for a, b in (("±", r"$\pm$"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
            text = text.replace(a, b)
        return text

    # Captions und Labels werden mitmaskiert -- ausser sie enthalten einen
    # Backslash, dann hat der Aufrufer bewusst eigenes LaTeX geschrieben.
    def esc_free(text):
        return text if "\\" in str(text) else esc(text)

    numeric = [pd.to_numeric(df[c], errors="coerce").notna().all() for c in df.columns]
    align = "".join("r" if n else "l" for n in numeric)

    lines = [r"\begin{table}[htbp]", r"  \centering"]
    if caption:
        lines.append(f"  \\caption{{{esc_free(caption)}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines += [r"  \small", f"  \\begin{{tabular}}{{{align}}}", r"    \toprule",
              "    " + " & ".join(esc(c) for c in df.columns) + r" \\", r"    \midrule"]

    previous = None
    for _, row in df.iterrows():
        if group_col and group_col in df.columns:
            if previous is not None and row[group_col] != previous:
                lines.append(r"    \midrule")
            previous = row[group_col]
        lines.append("    " + " & ".join(esc(row[c]) for c in df.columns) + r" \\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write(df: pd.DataFrame, path: str, fmt: str, **kwargs) -> None:
    """Schreibt ``df`` im gewuenschten Format nach ``path``."""
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "md":
        open(path, "w").write(to_markdown(df))
    elif fmt in ("tex", "latex"):
        open(path, "w").write(to_latex(df, **kwargs))
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise SystemExit(f"Unbekanntes Tabellenformat '{fmt}' (csv|md|tex|json)")
