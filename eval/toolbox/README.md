# MLflow-Auswertungs-Toolbox

Laedt Runs beliebiger MLflow-Experimente dieses Projekts, filtert sie und baut
daraus Tabellen (Terminal, CSV, Markdown, LaTeX) und Abbildungen (PNG/PDF/SVG).

```bash
python eval/mlflow_toolbox.py --help        # oder: python -m eval.toolbox --help
```

Standard-DB ist `runs/mlflow.db`, Standard-Ausgabeverzeichnis `eval/results/`.
Beides ueber `--db` bzw. `--outdir` aenderbar.

## Die fuenf Unterbefehle

| Befehl | Zweck |
|---|---|
| `experiments` | Welche Experimente liegen in der DB? |
| `columns` | Welche Spalten hat eine Auswahl (Params, Metriken, Tags)? |
| `table` | Gruppieren, aggregieren, exportieren |
| `plot` | Dieselbe Auswahl als Abbildung(en) |
| `report` | Tabelle *und* Abbildungen in einem Rutsch |

`table`, `plot` und `report` teilen sich denselben Auswahlpfad, d.h. jede
`--filter`/`--derive`/`--level`-Kombination wirkt in allen dreien gleich.

## Auswahl

```
-e, --experiment  Name(n) oder ID(s), kommasepariert
--level           parent | child | all
--status          FINISHED (Default) | FAILED | all
--derive          neue Spalte(n) per Regex
--filter          Filterausdruck (mehrfach = UND)
--query           zusaetzlicher pandas-query()-Ausdruck
--source          final (Default) | history
```

### `--level`: Parent oder Child

Die Sweeps loggen zweistufig: ein **Parent**-Run je Konfiguration mit den ueber
die Seeds aggregierten Metriken (`accuracy_mean`, `accuracy_std`, ...) und
darunter ein **Child**-Run je Seed mit den Rohwerten.

* `--level parent` ist schnell und nutzt die bereits geloggten Aggregate.
* `--level child` mittelt selbst ueber die Seeds. Das ist robuster, denn ein
  abgebrochener Parent-Run hat gar keine `*_mean`-Metriken -- im
  `mnist_bag_sweep` betrifft das vier Konfigurationen.

Modell- und Dataset-Parameter stehen nur am Parent. Bei `--level child` werden
sie automatisch an die Children durchgereicht (`--no-inherit` schaltet das ab).

### `--filter`

`spalte OP wert` mit `OP` aus `=`, `!=`, `>`, `>=`, `<`, `<=`, `~` (Regex),
`!~` (Regex negiert). Mehrfach angebbar, wirkt als UND. Bei `=`/`!=` ist eine
Komma-Werteliste erlaubt:

```bash
--filter "model=attention,gated_attention"   # Werteliste
--filter "num_bags>=200"                     # numerisch
--filter "run_name~^clam_.*k28$"             # Regex
```

Kurznamen (`model`) und lange Schreibweise (`params.model`) funktionieren beide;
die lange loest Namenskollisionen zwischen Param und Metrik auf.

### `--derive`

Zieht neue Spalten per Regex aus einer bestehenden -- jede Capture-Group fuellt
eine Zielspalte. Damit werden im Datasetnamen kodierte Sweep-Achsen erst
gruppierbar:

```bash
--derive 'num_bags,bag_length=dataset:bags(\d+)_len(\d+)'
```

### `--exclude-crashed`

Jeder Child-Run bekommt beim Laden die Spalte `crashed` (0/1). Ein Seed gilt als
abgestuerzt, wenn er **alle Test-Bags derselben Klasse** zuordnet:

* alle negativ: Precision = Recall = 0 und kein Bag ueberzaehlt (Bias = -MAE),
* alle positiv: Recall = 1 und Accuracy = Precision (= Anteil positiver Bags).

`--exclude-crashed` nimmt diese Seeds aus Tabellen und Abbildungen heraus; die
Tabelle zaehlt in `n` nur die verbleibenden Seeds und in `crashed` die
herausgenommenen. Die ausgeschlossenen Runs werden auf stderr gelistet. Ohne das
Flag laesst sich die Spalte normal filtern (`--filter "crashed=1"`).

### `--source history`

`search_runs` liefert nur den **zuletzt** geloggten Wert je Metrik. Fuer
Lernkurven laedt `--source history` die schrittweise Historie nach; die x-Achse
ist dann `step`.

## Tabellen

```bash
python eval/mlflow_toolbox.py table \
    -e mnist_bag_sweep --level child --status all \
    --derive 'num_bags,bag_length=dataset:bags(\d+)_len(\d+)' \
    --group-by bag_length,num_bags,model \
    --metrics accuracy,auc,patch_level_auc,count_mae \
    --out eval/results/sweep.tex --caption "MNIST-Bag-Sweep" --label tab:sweep
```

* `--agg` waehlt die Aggregate: `mean,std,median,min,max,count,sem,sum` sowie
  die robusten `q25,q75,iqr,mad`.
* Zusammengefasst wird in der Anzeige automatisch: `mean`+`std` zu `mean ± std`,
  `median`+`q25`+`q75` zu `median [q25, q75]`, `median`+`mad` zu `median ± mad`
  (`--raw` zeigt getrennte Spalten, CSV/JSON schreiben immer roh).
* `n` ist die Gruppengroesse. Eine Spalte `<metrik>_n` erscheint nur dort, wo
  eine Metrik weniger Werte hat als die Gruppe Runs -- also genau dann, wenn
  Werte fehlen.
* Das Zielformat kommt aus der Endung von `--out` (`.csv`, `.md`, `.tex`,
  `.json`) oder aus `--out-format`.

## Abbildungen

Vier Achsen bestimmen den Aufbau:

| Option | Wirkung |
|---|---|
| `--split` | eine eigene **Bilddatei** je Auspraegung; `--split metric` = eine Datei je Metrik |
| `--facet` | eine **Panel-Zeile** je Auspraegung |
| `--y` | eine **Panel-Spalte** je Metrik (ohne `--facet` umbrochen auf `--ncol`, Default 2) |
| `--series` | eine **Linie/Balkengruppe** je Auspraegung im Panel |

Sonderfall **eine Metrik je Abbildung**: bei genau einem `--y` werden die
Facetten nebeneinander umbrochen (statt untereinander gestapelt) und teilen sich
eine y-Achse -- so ist der Facetten-Vergleich direkt ablesbar:

```bash
--split metric --facet bag_length --series model --y accuracy,auc,count_mae
```

ergibt drei Dateien (`..._accuracy`, `..._auc`, `..._count_mae`), jede mit einem
Panel je Bag-Length und zwei Linien je Panel.

```bash
python eval/mlflow_toolbox.py plot \
    -e mnist_bag_sweep --level child --status all \
    --derive 'num_bags,bag_length=dataset:bags(\d+)_len(\d+)' \
    --split bag_length --kind line --x num_bags \
    --y accuracy,auc,patch_level_auc,count_mae --series model \
    --logy count_mae --format png,pdf --name sweep
```

`--kind`: `line`, `bar`, `scatter`, `box`, `heatmap`.
Beim Heatmap sind Zeilen = `--series`, Spalten = `--x`, Farbe = die Metrik.

Aggregiert wird immer aus den Roh-Runs ueber (`facet`, `x`, `series`): Linie bzw.
Balken = Lagemass, Band bzw. Fehlerbalken = Streuung. Ein Plot mit `--level child`
ist damit seed-aggregiert, ohne dass vorher eine Tabelle gebaut werden muss.

`--center mean|median` waehlt das Lagemass, `--err std|sem|iqr|none` die Streuung
(`--no-band` schaltet das Band ab). Der Default der Streuung richtet sich nach
dem Lagemass: `std` beim Mittelwert, `iqr` (Quartilsabstand) beim Median.

### Mittelwert oder Median?

Bei Seed-Sweeps kollabieren einzelne Laeufe auf Zufallsniveau. Die Verteilung
ueber die Seeds ist dann zweigipflig, und `mean ± std` beschreibt sie falsch --
das Band reicht bei beschraenkten Metriken sichtbar ueber 1,0 hinaus. `--center
median --err iqr` ist dagegen robust:

```bash
python eval/mlflow_toolbox.py plot -e mnist_bag_sweep_5seeds --level child \
    --derive 'num_bags,bag_length=dataset:bags(\d+)_len(\d+)' \
    --center median --facet bag_length --series model \
    --kind line --x num_bags --y accuracy --name robust
```

Soll der Median ueberall der Standard sein, statt ihn jedes Mal anzugeben:
in `cli.py` bei beiden `--center`-Argumenten `default="mean"` auf
`default="median"` aendern. Die Streuung zieht automatisch auf `iqr` nach.

`--logy` ohne Wert skaliert alle Panels logarithmisch, mit Wert nur die
genannten Metriken (`--logy count_mae`) -- Metriken auf [0,1] gehoeren nicht auf
eine Log-Achse.

### Farben

Feste, farbenblind-sichere Reihenfolge, nie zyklisch wiederverwendet. Linienstil und Marker
codieren die Serie zusaetzlich, damit Identitaet nicht allein an der Farbe
haengt. Mehr als 8 Serien werden abgelehnt -- dann ist `--facet` oder `--split`
das richtige Mittel. Ab 4 Serien warnt `scatter`/`heatmap`.

## Beispiele

```bash
# Was liegt in der DB, was hat ein Experiment fuer Spalten?
python eval/mlflow_toolbox.py experiments
python eval/mlflow_toolbox.py columns -e clam_sweep_v3 --art param
python eval/mlflow_toolbox.py columns -e mnist_bag_sweep --level child --grep auc

# CLAM-Sweep als Heatmap ueber zwei Hyperparameter
python eval/mlflow_toolbox.py plot -e clam_sweep_v3 --level parent \
    --kind heatmap --x clam_k_sample --y auc_mean,patch_level_auc_mean \
    --series clam_bag_weight --name clam_grid

# Streuung ueber die Seeds statt Mittelwert
python eval/mlflow_toolbox.py plot -e mnist_bag_sweep --level child \
    --derive 'num_bags,bag_length=dataset:bags(\d+)_len(\d+)' \
    --kind box --x num_bags --y patch_level_auc --series model --name streuung

# Lernkurven
python eval/mlflow_toolbox.py plot -e mnist_bag_sweep --level child \
    --source history --kind line --x step --y train_loss,val_loss \
    --series model --no-band --name kurven

# Tabelle + alle Abbildungen in einem Aufruf
python eval/mlflow_toolbox.py report -e mnist_bag_sweep --level child --status all \
    --derive 'num_bags,bag_length=dataset:bags(\d+)_len(\d+)' \
    --group-by bag_length,num_bags,model \
    --metrics accuracy,auc,patch_level_auc,count_mae \
    --x num_bags --series model --split bag_length --logy count_mae --name sweep
```

## Als Bibliothek

Fuer Sonderfaelle, die die CLI nicht abdeckt:

```python
import sys; sys.path.insert(0, "eval")
from toolbox import loading, tables, plots

df = loading.load_runs("runs/mlflow.db", ["mnist_bag_sweep"], level="child")
df = loading.derive(df, [r"num_bags,bag_length=dataset:bags(\d+)_len(\d+)"])
df = loading.apply_filters(df, ["num_bags>=100"])

tab = tables.aggregate(df, ["model"], ["accuracy", "auc"])
fig = plots.make_figure(df, "line", "num_bags", ["accuracy"], series="model")
plots.save(fig, "eval/results/meins", ("png", "pdf"))
```

## Aufbau

| Datei | Inhalt |
|---|---|
| `loading.py` | Verbindung, Flachklopfen, Parent/Child, `--derive`, Filter, Historie |
| `tables.py` | Aggregation, `mean ± std`-Formatierung, CSV/MD/LaTeX/JSON |
| `plots.py` | Panel-Grammatik, die fuenf Plot-Typen, Palette |
| `cli.py` | Argumente und Unterbefehle |
