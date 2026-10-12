"""Plot-Generator: eine kleine Grammatik statt eines Skripts je Abbildung.

Vier Achsen bestimmen den Aufbau einer Abbildung::

    --split   eine eigene Bilddatei je Auspraegung
    --facet   eine Panel-Zeile je Auspraegung
    --y       eine Panel-Spalte je Metrik
    --series  eine Linie/Balkengruppe je Auspraegung innerhalb des Panels

Aggregiert wird immer aus den Roh-Runs ueber (facet, x, series): Linie/Balken =
Mittelwert, Band/Fehlerbalken = Streuung.  Damit ist ein Plot mit ``--level child``
seed-gemittelt, ohne dass vorher eine Tabelle gebaut werden muss.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Palette (validierte Reihenfolge, wird nie zyklisch wiederverwendet).         #
# Identisch zu eval/scripts/plot_clam_sweep.py und plot_bars.py.               #
# --------------------------------------------------------------------------- #
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
#: Zweitcodierung, damit Identitaet nie allein an der Farbe haengt (S/W-Druck, CVD).
DASHES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (4, 2, 1, 2))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d9d6"
#: Sequenzielle Rampe: ein Farbton, hell -> dunkel (nie ein Regenbogen).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues_1hue", ["#f2f6fc", "#9cc2ee", "#2a78d6", "#123a68"])

KINDS = ("line", "bar", "scatter", "box", "heatmap")

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "figure.dpi": 130,
    "savefig.dpi": 200, "savefig.facecolor": "white",
})


def _clean(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6, labelsize=9)


def _levels(df, col):
    """Sortierte Auspraegungen einer Spalte (numerisch, wenn moeglich)."""
    if col is None:
        return [None]
    vals = df[col].dropna().unique()
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


def _check_series(levels, kind):
    if len(levels) > len(PALETTE):
        raise SystemExit(
            f"{len(levels)} Serien, aber die Palette hat nur {len(PALETTE)} "
            f"unterscheidbare Farben. Filtere enger oder nutze --facet/--split.")
    if kind in ("scatter", "heatmap") and len(levels) > 3:
        print(f"  Hinweis: {len(levels)} Serien in einem {kind}-Plot sind farblich "
              f"nicht mehr sicher trennbar (Empfehlung: hoechstens 3).")


def _agg(df, x, y, series, err, center="mean"):
    """Lage und Streuung je (series, x) fuer eine Metrik.

    ``center`` waehlt Mittelwert oder Median. Die Streuung kommt als explizite
    Unter-/Obergrenze zurueck (``_lo``/``_hi``), weil der IQR um den Median
    asymmetrisch liegt -- ein einzelner ±-Wert koennte ihn nicht abbilden.
    """
    keys = [k for k in (series, x) if k is not None]
    frame = df.groupby(keys, dropna=False)[y] if keys else df[y]

    if keys:
        mid = frame.median() if center == "median" else frame.mean()
        out = mid.to_frame(y)
        if err == "iqr":
            out["_lo"], out["_hi"] = frame.quantile(0.25), frame.quantile(0.75)
        elif err in ("std", "sem"):
            spread = frame.std() if err == "std" else frame.sem()
            out["_lo"], out["_hi"] = mid - spread, mid + spread
        else:
            out["_lo"] = out["_hi"] = np.nan
        out["_n"] = frame.count()
        return out.reset_index()

    mid = frame.median() if center == "median" else frame.mean()
    if err == "iqr":
        lo, hi = frame.quantile(0.25), frame.quantile(0.75)
    elif err in ("std", "sem"):
        spread = frame.std() if err == "std" else frame.sem()
        lo, hi = mid - spread, mid + spread
    else:
        lo = hi = np.nan
    return pd.DataFrame({y: [mid], "_lo": [lo], "_hi": [hi], "_n": [frame.count()]})


# --------------------------------------------------------------------------- #
# Panel-Typen                                                                  #
# --------------------------------------------------------------------------- #
def _panel_line(ax, data, x, y, series, levels, err, band, center="mean"):
    for i, lvl in enumerate(levels):
        sub = data if lvl is None else data[data[series] == lvl]
        agg = _agg(sub, x, y, None, err, center).sort_values(x)
        xs, ys = agg[x].to_numpy(), agg[y].to_numpy()
        lo, hi = agg["_lo"].to_numpy(), agg["_hi"].to_numpy()
        ok = ~pd.isna(ys)
        color = PALETTE[i]
        if band and not np.all(pd.isna(lo)):
            ax.fill_between(xs[ok], np.where(pd.isna(lo), ys, lo)[ok],
                            np.where(pd.isna(hi), ys, hi)[ok], color=color,
                            alpha=0.13, lw=0)
        ax.plot(xs[ok], ys[ok], ls=DASHES[i], color=color, lw=1.8, zorder=3)
        ax.plot(xs[ok], ys[ok], ls="none", marker=MARKERS[i], ms=6, color=color,
                mec="white", mew=1.1, zorder=4)


def _panel_bar(ax, data, x, y, series, levels, err, _band, center="mean"):
    cats = _levels(data, x)
    pos = np.arange(len(cats), dtype=float)
    width = 0.8 / max(len(levels), 1)
    for i, lvl in enumerate(levels):
        sub = data if lvl is None else data[data[series] == lvl]
        agg = _agg(sub, x, y, None, err, center).set_index(x)
        vals = np.array([agg[y].get(c, np.nan) for c in cats], dtype=float)
        lo = np.array([agg["_lo"].get(c, np.nan) for c in cats], dtype=float)
        hi = np.array([agg["_hi"].get(c, np.nan) for c in cats], dtype=float)
        # yerr erwartet Abstaende zum Balkenende, nicht absolute Grenzen.
        errs = np.vstack([np.nan_to_num(vals - lo), np.nan_to_num(hi - vals)])
        offset = (i - (len(levels) - 1) / 2) * width
        ax.bar(pos + offset, vals, width * 0.92, color=PALETTE[i], zorder=3,
               yerr=None if np.all(pd.isna(lo)) else errs,
               error_kw={"ecolor": MUTED, "elinewidth": 0.9, "capsize": 2.5})
    ax.set_xticks(pos)
    ax.set_xticklabels([str(c) for c in cats])


def _panel_scatter(ax, data, x, y, series, levels, _err, _band, center="mean"):
    for i, lvl in enumerate(levels):
        sub = data if lvl is None else data[data[series] == lvl]
        ax.scatter(sub[x], sub[y], s=42, color=PALETTE[i], marker=MARKERS[i],
                   edgecolors="white", linewidths=1.0, zorder=3, alpha=0.9)


def _panel_box(ax, data, x, y, series, levels, _err, _band, center="mean"):
    cats = _levels(data, x)
    width = 0.8 / max(len(levels), 1)
    for i, lvl in enumerate(levels):
        sub = data if lvl is None else data[data[series] == lvl]
        groups = [sub[sub[x] == c][y].dropna().to_numpy() for c in cats]
        offset = (i - (len(levels) - 1) / 2) * width
        bp = ax.boxplot(groups, positions=np.arange(len(cats)) + offset,
                        widths=width * 0.85, patch_artist=True, manage_ticks=False,
                        medianprops={"color": INK, "lw": 1.2},
                        flierprops={"marker": ".", "ms": 4,
                                    "markerfacecolor": MUTED, "markeredgecolor": "none"})
        for patch in bp["boxes"]:
            patch.set(facecolor=PALETTE[i], alpha=0.55, edgecolor=PALETTE[i], lw=1.1)
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set(color=PALETTE[i], lw=1.0)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([str(c) for c in cats])


def _panel_heatmap(ax, data, x, y, series, levels, _err, _band, center="mean"):
    """Zeilen = Serien, Spalten = x, Farbe = Metrik (sequenziell, einfarbig)."""
    cats = _levels(data, x)
    grid = np.full((len(levels), len(cats)), np.nan)
    for r, lvl in enumerate(levels):
        sub = data if lvl is None else data[data[series] == lvl]
        grouped = sub.groupby(x, dropna=False)[y]
        agg = grouped.median() if center == "median" else grouped.mean()
        for c, cat in enumerate(cats):
            grid[r, c] = agg.get(cat, np.nan)

    im = ax.imshow(grid, cmap=SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([str(c) for c in cats])
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(["" if l is None else str(l) for l in levels])
    ax.grid(False)
    span = np.nanmax(grid) - np.nanmin(grid) if np.isfinite(grid).any() else 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if np.isnan(grid[r, c]):
                continue
            light = span and (grid[r, c] - np.nanmin(grid)) / span > 0.6
            ax.text(c, r, f"{grid[r, c]:.3g}", ha="center", va="center", fontsize=8.5,
                    color="white" if light else INK)
    if series:
        ax.set_ylabel(series, fontsize=9.5, color=MUTED)
    ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03).outline.set_visible(False)


_PANELS = {"line": _panel_line, "bar": _panel_bar, "scatter": _panel_scatter,
           "box": _panel_box, "heatmap": _panel_heatmap}


# --------------------------------------------------------------------------- #
# Abbildung                                                                    #
# --------------------------------------------------------------------------- #
def make_figure(df, kind, x, ys, series=None, facet=None, err="std", band=True,
                logy=(), sharey=True, ylim=None, title=None, panel_size=(3.6, 2.6),
                ncol=None, center="mean"):
    """Baut eine Abbildung aus Panels.

    Mit ``facet``: eine Panel-Zeile je Auspraegung, eine Spalte je Metrik in ``ys``
    (die y-Achse wird dann spaltenweise geteilt, damit die Facetten vergleichbar
    sind).  Ohne ``facet``: die Metriken werden in ``ncol`` Spalten umbrochen.

    ``logy`` ist die Menge der Metriken mit logarithmischer y-Achse.
    """
    if kind not in _PANELS:
        raise SystemExit(f"Unbekanntes --kind '{kind}' ({'|'.join(KINDS)})")
    for col in [c for c in (x, series, facet) if c]:
        if col not in df.columns:
            raise SystemExit(f"Spalte '{col}' gibt es nicht. 'columns' zeigt sie an.")
    missing = [y for y in ys if y not in df.columns]
    if missing:
        raise SystemExit(f"Unbekannte Metriken: {', '.join(missing)}")

    slevels = _levels(df, series)
    _check_series(slevels, kind)
    flevels = _levels(df, facet)

    # Panel-Raster: mit facet -> Zeilen = Facetten, Spalten = Metriken.
    # Ohne facet -> Metriken in ncol Spalten umbrechen.
    if facet and len(ys) == 1:
        # Eine Metrik, viele Facetten -> Facetten umbrechen statt untereinander
        # stapeln. Die y-Achse wird geteilt, damit die Facetten vergleichbar sind.
        ncols = ncol or min(len(flevels), 3)
        nrows = -(-len(flevels) // ncols)
        cells = [(i // ncols, i % ncols, lvl, ys[0]) for i, lvl in enumerate(flevels)]
        share_y = "all" if sharey else "none"
    elif facet:
        ncols, nrows = len(ys), len(flevels)
        cells = [(r, c, flevels[r], ys[c]) for r in range(nrows) for c in range(ncols)]
        share_y = "col" if sharey else "none"
    else:
        ncols = ncol or min(len(ys), 2)
        nrows = -(-len(ys) // ncols)
        cells = [(i // ncols, i % ncols, None, y) for i, y in enumerate(ys)]
        share_y = "none"          # verschiedene Metriken teilen keine Skala

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             sharex="all" if kind != "heatmap" else "none",
                             sharey=share_y,
                             figsize=(panel_size[0] * ncols + 0.8,
                                      panel_size[1] * nrows + 0.7))
    for ax in axes.ravel():
        ax.set_visible(False)

    for r, c, flvl, y in cells:
        rows = df if flvl is None else df[df[facet] == flvl]
        ax = axes[r][c]
        ax.set_visible(True)
        if True:
            _PANELS[kind](ax, rows, x, y, series, slevels, err, band, center)
            if kind != "heatmap" and (y in logy or logy == "all"):
                ax.set_yscale("log")
            if ylim and kind != "heatmap":
                ax.set_ylim(*ylim)
            if kind != "heatmap":
                _clean(ax)
            # Mit Facetten UND mehreren Metriken benennt der Panel-Titel die
            # Metrik (Spalte) und die y-Achse die Facette (Zeile) -- sonst stuende
            # in jedem Panel derselbe Facettenname.
            # Ohne Facetten werden die Metriken auf Zeilen umbrochen -> jedes Panel
            # braucht seinen eigenen Titel.
            if flvl is None:
                head, show = y, True
            elif len(ys) > 1:
                head, show = y, (r == 0)
            else:
                head, show = f"{facet} = {flvl}", True
            if show:
                ax.set_title(head, fontsize=10, color=INK, loc="left", pad=6)
            if r == nrows - 1:
                ax.set_xlabel(x)
            if c == 0 and flvl is not None:
                # Bei einer Metrik benennt die y-Achse die Metrik, sonst die Facette.
                ax.set_ylabel(y if len(ys) == 1 else f"{facet} = {flvl}",
                              fontsize=9.5, color=INK)

    # Bei Sweep-Daten sind die gemessenen x-Werte die richtigen Ticks -- sonst
    # laesst der Auto-Locator Stuetzstellen am Rand unbeschriftet.
    if kind in ("line", "scatter") and pd.api.types.is_numeric_dtype(df[x]):
        xs = sorted(df[x].dropna().unique())
        if 1 < len(xs) <= 12:
            for ax in axes.ravel():
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{v:g}" for v in xs])
        elif all(float(v).is_integer() for v in xs):
            # Ganzzahlige x-Werte (Epochen, Schritte) nie als 2.5 beschriften.
            for ax in axes.ravel():
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    # Bei umbrochenem Raster hat die letzte Zeile evtl. Luecken -> unterstes
    # sichtbares Panel je Spalte bekommt die x-Beschriftung und Ticks.
    for c in range(ncols):
        visible = [r for r in range(nrows) if axes[r][c].get_visible()]
        if visible:
            ax = axes[visible[-1]][c]
            ax.set_xlabel(x)
            ax.tick_params(labelbottom=True)

    # Titel und Legende erst nach dem Layout setzen und direkt ueber die Achsen
    # legen (Titel links, Legende rechts auf derselben Zeile) -- sonst haengen sie
    # am Figurenrand und lassen ein Loch, das mit der Panelzahl waechst.
    has_legend = bool(series) and len(slevels) > 1 and kind != "heatmap"
    band = 0.34 if (has_legend or title) else 0.0
    height = panel_size[1] * nrows + 0.7
    fig.tight_layout(rect=(0, 0, 1, 1 - band / height))

    # tightbbox statt get_position: die Panel-Titel stehen ueber der Achsenbox und
    # muessen mitgezaehlt werden, sonst schreibt die Legende darueber.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    top = max(ax.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).y1
              for ax in axes.ravel() if ax.get_visible())

    if has_legend:
        if kind == "line":
            handles = [Line2D([], [], color=PALETTE[i], ls=DASHES[i], marker=MARKERS[i],
                              ms=6, mec="white", mew=1.1, lw=1.8, label=str(l))
                       for i, l in enumerate(slevels)]
        elif kind == "scatter":
            handles = [Line2D([], [], color=PALETTE[i], ls="none", marker=MARKERS[i],
                              ms=7, mec="white", mew=1.0, label=str(l))
                       for i, l in enumerate(slevels)]
        else:
            handles = [matplotlib.patches.Patch(facecolor=PALETTE[i], label=str(l))
                       for i, l in enumerate(slevels)]
        fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, top + 0.010),
                   ncol=min(len(slevels), 4), fontsize=9, labelcolor=MUTED,
                   title=series, title_fontsize=9, alignment="right")
    if title:
        fig.suptitle(title, fontsize=11.5, color=INK, x=0.008, ha="left",
                     y=top + 0.012, va="bottom")

    return fig


def save(fig, stem, formats=("png",)):
    paths = []
    for ext in formats:
        path = f"{stem}.{ext}"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
