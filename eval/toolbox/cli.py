"""Kommandozeile der MLflow-Auswertungs-Toolbox.

Fuenf Unterbefehle:

    experiments   welche Experimente liegen in der DB?
    columns       welche Spalten hat ein Experiment (Params, Metriken, Tags)?
    table         gefiltert laden, gruppieren, aggregieren, exportieren
    plot          dieselbe Auswahl als Abbildung(en)
    report        Tabelle + Abbildungen in einem Rutsch in ein Ausgabeverzeichnis
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

if __package__ in (None, ""):                      # Direktaufruf der Datei
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from toolbox import loading, plots, tables
else:
    from . import loading, plots, tables

DEFAULT_DB = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "mlflow.db"))
DEFAULT_OUT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

EPILOG = """\
Beispiele
---------
  # Was liegt in der DB?
  %(prog)s experiments
  %(prog)s columns -e mnist_bag_sweep --level child

  # Tabelle: Seeds mitteln, nach Modell und Bag-Groesse gruppieren
  %(prog)s table -e mnist_bag_sweep --level child \\
      --derive "num_bags,bag_length=dataset:bags(\\d+)_len(\\d+)" \\
      --group-by bag_length,num_bags,model \\
      --metrics accuracy,auc,patch_level_auc,count_mae \\
      --out-format tex --out eval/results/sweep.tex

  # Nur gated, ab 100 Bags, als Abbildung je Bag-Length
  %(prog)s plot -e mnist_bag_sweep --level child \\
      --derive "num_bags,bag_length=dataset:bags(\\d+)_len(\\d+)" \\
      --filter "num_bags>=100" --split bag_length \\
      --kind line --x num_bags --y accuracy,auc,patch_level_auc,count_mae \\
      --series model --name sweep

  # Lernkurven aus der Metrik-Historie
  %(prog)s plot -e clam_sweep_v3 --source history \\
      --kind line --x step --y train_loss,val_loss --series run_name
"""


# --------------------------------------------------------------------------- #
# Argumente                                                                    #
# --------------------------------------------------------------------------- #
def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def _add_selection(p):
    g = p.add_argument_group("Auswahl")
    g.add_argument("-e", "--experiment", required=True, type=_csv,
                   help="Experimentname(n) oder -ID(s), kommasepariert")
    g.add_argument("--level", default="parent", choices=("parent", "child", "all"),
                   help="parent = aggregierte Konfigurationen (Default), "
                        "child = ein Run je Seed, all = beides")
    g.add_argument("--status", default="FINISHED",
                   help="Run-Status, 'all' fuer alle (Default: FINISHED)")
    g.add_argument("--no-inherit", action="store_true",
                   help="Params bei --level child NICHT vom Parent erben")
    g.add_argument("--derive", action="append", metavar="SPEC",
                   help="Neue Spalte(n) per Regex: 'ziel[,ziel]=quelle:regex' "
                        "(mehrfach angebbar)")
    g.add_argument("--filter", action="append", metavar="EXPR",
                   help="Filter 'spalte OP wert', OP aus = != > >= < <= ~ !~ "
                        "(~ = Regex; bei = / != ist eine Komma-Werteliste erlaubt). "
                        "Mehrfach angebbar, wirkt als UND.")
    g.add_argument("--query", help="Zusaetzlicher pandas-query()-Ausdruck")
    g.add_argument("--exclude-crashed", action="store_true",
                   help="Abgestuerzte Seeds (alle Test-Bags einer Klasse zugeordnet) "
                        "aus Kennzahlen und Abbildungen herausnehmen; die Tabelle "
                        "zaehlt sie in der Spalte 'crashed'")
    g.add_argument("--source", default="final", choices=("final", "history"),
                   help="final = zuletzt geloggter Wert je Metrik (Default), "
                        "history = schrittweise Historie (fuer Lernkurven)")
    g.add_argument("--history-keys", type=_csv,
                   help="Metriken fuer --source history (Default: die aus --y/--metrics)")


def _add_db(p):
    p.add_argument("--db", default=DEFAULT_DB,
                   help=f"MLflow-SQLite-Datei oder Tracking-URI (Default: {DEFAULT_DB})")


def build_parser():
    p = argparse.ArgumentParser(
        prog="mlflow_toolbox", description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("experiments", help="Experimente auflisten")
    _add_db(e)

    c = sub.add_parser("columns", help="Verfuegbare Spalten eines Experiments zeigen")
    _add_db(c)
    _add_selection(c)
    c.add_argument("--art", type=_csv, help="Nur diese Arten: param,metrik,tag,meta")
    c.add_argument("--grep", help="Nur Spalten, deren Name diesen Regex enthaelt")

    t = sub.add_parser("table", help="Tabelle bauen und exportieren")
    _add_db(t)
    _add_selection(t)
    t.add_argument("--group-by", type=_csv, help="Gruppierungsspalten")
    t.add_argument("--metrics", type=_csv, required=True, help="Auszuwertende Metriken")
    t.add_argument("--agg", type=_csv, default=["mean", "std"],
                   help=f"Aggregate ({'|'.join(tables.AGGS)}), Default: mean,std")
    t.add_argument("--sort", type=_csv, help="Sortierspalten der Ergebnistabelle")
    t.add_argument("--precision", type=int, default=3, help="Nachkommastellen (Default: 3)")
    t.add_argument("--raw", action="store_true",
                   help="Rohe Zahlenspalten statt 'mean ± std'")
    t.add_argument("--out", help="Zieldatei (ohne Angabe nur Ausgabe im Terminal)")
    t.add_argument("--out-format", default=None, choices=("csv", "md", "tex", "json"),
                   help="Default: aus der Dateiendung von --out")
    t.add_argument("--caption", default="", help="LaTeX-Caption")
    t.add_argument("--label", default="", help="LaTeX-Label")

    g = sub.add_parser("plot", help="Abbildung(en) erzeugen")
    _add_db(g)
    _add_selection(g)
    g.add_argument("--kind", default="line", choices=plots.KINDS)
    g.add_argument("--x", required=True, help="x-Achse")
    g.add_argument("--y", type=_csv, required=True,
                   help="Metrik(en); jede wird eine Panel-Spalte")
    g.add_argument("--series", help="Spalte, die Linien/Balken innerhalb eines Panels trennt")
    g.add_argument("--facet", help="Spalte, die Panel-Zeilen aufspannt")
    g.add_argument("--split", help="Spalte, die eine eigene Bilddatei je Wert erzeugt; "
                                   "der Sonderwert 'metric' erzeugt eine Datei je --y-Metrik")
    g.add_argument("--center", default="mean", choices=("mean", "median"),
                   help="Lagemass fuer Linie/Balken (Default: mean). 'median' ist "
                        "robust gegen kollabierte Seeds.")
    g.add_argument("--err", default=None, choices=("std", "sem", "iqr", "none"),
                   help="Streuung fuer Band/Fehlerbalken. Default: std beim "
                        "Mittelwert, iqr (Quartilsabstand) beim Median.")
    g.add_argument("--no-band", action="store_true", help="Kein Streuungsband bei --kind line")
    g.add_argument("--logy", nargs="?", const="all", default=None, metavar="METRIKEN",
                   help="Logarithmische y-Achse: ohne Wert fuer alle Panels, sonst "
                        "kommaseparierte Metriken (z.B. --logy count_mae)")
    g.add_argument("--ncol", type=int,
                   help="Panel-Spalten, wenn kein --facet gesetzt ist (Default: 2)")
    g.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"))
    g.add_argument("--no-sharey", action="store_true")
    g.add_argument("--title", help="Titel (bei --split ergaenzt um den Split-Wert)")
    g.add_argument("--no-title", action="store_true",
                   help="Keinen Titel ins Bild setzen, auch nicht den automatischen "
                        "Split-Titel -- fuer Abbildungen, die eine LaTeX-Caption tragen")
    g.add_argument("--outdir", default=DEFAULT_OUT, help=f"Default: {DEFAULT_OUT}")
    g.add_argument("--name", help="Dateiname-Stamm (Default: aus Experiment und Metriken)")
    g.add_argument("--format", type=_csv, default=["png"], help="png,pdf,svg (Default: png)")
    g.add_argument("--panel-size", type=float, nargs=2, default=(3.6, 2.6),
                   metavar=("B", "H"), help="Panelgroesse in Zoll (Default: 3.6 2.6)")

    r = sub.add_parser("report", help="Tabelle und Abbildungen in einem Rutsch")
    _add_db(r)
    _add_selection(r)
    r.add_argument("--group-by", type=_csv, required=True)
    r.add_argument("--metrics", type=_csv, required=True)
    r.add_argument("--x", required=True)
    r.add_argument("--series")
    r.add_argument("--facet", help="Spalte, die Panel-Zeilen aufspannt")
    r.add_argument("--split", help="Wie bei 'plot'; 'metric' = eine Datei je Metrik")
    r.add_argument("--kind", default="line", choices=plots.KINDS)
    r.add_argument("--center", default="mean", choices=("mean", "median"),
                   help="Lagemass fuer Tabelle und Plots (Default: mean)")
    r.add_argument("--logy", nargs="?", const="all", default=None, metavar="METRIKEN",
                   help="Wie bei 'plot': ohne Wert alle Panels, sonst Metrikliste")
    r.add_argument("--outdir", default=DEFAULT_OUT)
    r.add_argument("--name", required=True, help="Stamm fuer alle erzeugten Dateien")
    r.add_argument("--format", type=_csv, default=["png", "pdf"])
    r.add_argument("--precision", type=int, default=3)
    return p


# --------------------------------------------------------------------------- #
# Gemeinsamer Ladepfad                                                         #
# --------------------------------------------------------------------------- #
def select(args, history_keys=None, carry=None):
    """Laden -> abgeleitete Spalten -> Filter; identisch fuer alle Unterbefehle."""
    df = loading.load_runs(args.db, args.experiment, level=args.level,
                           inherit_params=not args.no_inherit, status=args.status)
    df = loading.derive(df, args.derive)
    df = loading.apply_filters(df, args.filter, args.query)
    if getattr(args, "exclude_crashed", False):
        _report_crashed(df)

    if getattr(args, "source", "final") == "history":
        keys = args.history_keys or history_keys or []
        keys = [k for k in keys if k != "step"]
        if not keys:
            raise SystemExit("--source history braucht --history-keys (oder --y/--metrics).")
        df = loading.load_history(args.db, df, keys, carry=carry)
    return df


def _report_crashed(df):
    """Listet die abgestuerzten Seeds der Auswahl, damit der Ausschluss nachpruefbar ist."""
    hit = df[df["crashed"] == 1]
    print(f"{len(hit)} abgestuerzte Seed(s) ausgeschlossen", file=sys.stderr)
    for _, r in hit.iterrows():
        print(f"  {r.get('parent_run_name', '')} / {r['run_name']}", file=sys.stderr)


def _without_crashed(args, df):
    """Fuer Abbildungen: abgestuerzte Seeds vor dem Zeichnen entfernen."""
    if getattr(args, "exclude_crashed", False):
        return df[df["crashed"] == 0]
    return df


def _err(args):
    """Passendes Streuungsmass: Std zum Mittelwert, IQR zum Median."""
    if args.err:
        return args.err
    return "iqr" if getattr(args, "center", "mean") == "median" else "std"


def _logy(value):
    """``--logy`` ohne Wert -> alle Panels, sonst die genannten Metriken."""
    if value is None:
        return ()
    return "all" if value == "all" else _csv(value)


def _splits(args, df):
    """Zerlegt die Auswahl in (Daten, Metriken, Titel, Dateinamen-Suffix) je Abbildung.

    ``--split metric`` erzeugt eine Abbildung je Metrik, jede andere Angabe eine
    Abbildung je Auspraegung der genannten Spalte.
    """
    metrics = args.y if hasattr(args, "y") else args.metrics
    base = None if getattr(args, "no_title", False) else getattr(args, "title", None)
    if not args.split:
        return [(df, metrics, base, "")]

    quiet = getattr(args, "no_title", False)
    if args.split == "metric" and "metric" not in df.columns:
        return [(df, [m], None if quiet else (f"{base} — {m}" if base else m), f"_{m}")
                for m in metrics]

    out = []
    for value in plots._levels(df, args.split):
        title = f"{args.split} = {value}"
        out.append((df[df[args.split] == value], metrics,
                    None if quiet else (f"{base} — {title}" if base else title),
                    f"_{args.split}{value}"))
    return out


def _stem(args, metrics):
    if args.name:
        return args.name
    return "_".join(args.experiment) + "__" + "_".join(metrics[:3])


# --------------------------------------------------------------------------- #
# Unterbefehle                                                                 #
# --------------------------------------------------------------------------- #
def cmd_experiments(args):
    print(loading.list_experiments(args.db).to_string(index=False))


def cmd_columns(args):
    df = select(args)
    overview = loading.column_overview(df)
    if args.art:
        overview = overview[overview["art"].isin(args.art)]
    if args.grep:
        overview = overview[overview["spalte"].str.contains(args.grep, regex=True)]
    if not len(overview):
        raise SystemExit("Keine Spalte passt zur Auswahl.")
    print(f"{len(df)} Runs, {len(overview)} Spalten\n")
    print(overview.to_string(index=False))


def cmd_table(args):
    df = select(args, history_keys=args.metrics, carry=args.group_by)
    result = tables.aggregate(df, args.group_by, args.metrics, args.agg,
                              exclude_crashed=args.exclude_crashed)
    if args.sort:
        missing = [s for s in args.sort if s not in result.columns]
        if missing:
            raise SystemExit(f"--sort: unbekannte Spalten {', '.join(missing)}")
        result = result.sort_values(args.sort).reset_index(drop=True)

    shown = result if args.raw else tables.render(result, args.precision)
    print(tables.to_text(shown))

    if args.out:
        fmt = args.out_format or os.path.splitext(args.out)[1].lstrip(".").lower() or "csv"
        fmt = {"latex": "tex"}.get(fmt, fmt)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        payload = result if (fmt in ("csv", "json") and not args.raw) else shown
        kwargs = {}
        if fmt == "tex":
            kwargs = {"caption": args.caption, "label": args.label,
                      "group_col": args.group_by[0] if args.group_by else None}
        tables.write(payload, args.out, fmt, **kwargs)
        print(f"\ngeschrieben: {args.out}")


def cmd_plot(args):
    df = select(args, history_keys=args.y,
                carry=[c for c in (args.series, args.facet, args.split) if c])
    df = _without_crashed(args, df)
    stem = _stem(args, args.y)
    written = []
    for rows, ys, title, suffix in _splits(args, df):
        fig = plots.make_figure(
            rows, args.kind, args.x, ys, series=args.series, facet=args.facet,
            err=_err(args), band=not args.no_band, logy=_logy(args.logy),
            sharey=not args.no_sharey, ylim=tuple(args.ylim) if args.ylim else None,
            title=title, panel_size=tuple(args.panel_size), ncol=args.ncol,
            center=args.center)
        written += plots.save(fig, os.path.join(args.outdir, stem + suffix), args.format)
    for path in written:
        print("geschrieben:", path)


def cmd_report(args):
    args.source, args.history_keys = "final", None
    df = select(args)

    os.makedirs(args.outdir, exist_ok=True)
    aggs = ("median", "q25", "q75") if args.center == "median" else ("mean", "std")
    result = tables.aggregate(df, args.group_by, args.metrics, aggs,
                              exclude_crashed=args.exclude_crashed)
    shown = tables.render(result, args.precision)
    print(tables.to_text(shown))
    df = _without_crashed(args, df)

    base = os.path.join(args.outdir, args.name)
    result.to_csv(base + ".csv", index=False)
    open(base + ".md", "w").write(tables.to_markdown(shown))
    open(base + ".tex", "w").write(tables.to_latex(
        shown, caption=f"Ergebnisse aus MLflow-Experiment {', '.join(args.experiment)}.",
        label=f"tab:{args.name}", group_col=args.group_by[0]))
    print(f"\ngeschrieben: {base}.csv / .md / .tex")

    args.title = None
    for rows, ys, title, suffix in _splits(args, df):
        fig = plots.make_figure(rows, args.kind, args.x, ys, series=args.series,
                                facet=args.facet, logy=_logy(args.logy),
                                title=title or args.name, center=args.center,
                                err="iqr" if args.center == "median" else "std")
        for path in plots.save(fig, base + suffix, args.format):
            print("geschrieben:", path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    pd.set_option("display.width", 220, "display.max_columns", 80,
                  "display.max_rows", 400)
    {"experiments": cmd_experiments, "columns": cmd_columns, "table": cmd_table,
     "plot": cmd_plot, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
