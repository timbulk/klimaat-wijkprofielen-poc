# klimaat-wijkprofielen-poc

> **Proof of concept** — Enrich CBS neighbourhood polygons with climate impact data from the
> Klimaateffectatlas by calculating zonal statistics per neighbourhood or district.

---

## Goal

Dutch municipalities and researchers often need to understand *which neighbourhoods are most affected by climate risks* such as heat stress, flooding, or drought. This tool bridges two authoritative Dutch open datasets:

| Dataset | Source | Format |
|---|---|---|
| Wijk- en Buurtenkaart | [CBS / PDOK](https://www.pdok.nl/downloads/-/article/cbs-wijk-en-buurtkaart) | GeoPackage / Shapefile |
| Klimaateffectatlas rasters | [Klimaateffectatlas](https://www.klimaateffectatlas.nl) | GeoTIFF |

For each neighbourhood polygon the tool calculates **zonal statistics** (mean, max, min, std, percentage affected) from one or more climate rasters and writes the result to a new GeoPackage ready for GIS or further analysis.

---

## Prerequisites

- Python 3.10+
- GDAL system libraries (required by `rasterio` / `fiona`)

### macOS
```bash
brew install gdal
```

### Ubuntu / Debian
```bash
sudo apt-get install gdal-bin libgdal-dev
```

### Windows
Install [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or use the Conda approach below.

> **Tip:** Using [Conda / Mamba](https://docs.conda.io/) avoids most GDAL headaches:
> ```bash
> conda create -n klimaat python=3.11 geopandas rasterio rasterstats fiona pyogrio matplotlib
> conda activate klimaat
> ```

---

## Installation

```bash
git clone https://github.com/timbulk/klimaat-wijkprofielen-poc.git
cd klimaat-wijkprofielen-poc

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## How to get the data

### 1. CBS Wijk- en Buurtenkaart

Download the latest edition from PDOK or CBS:

```
https://www.pdok.nl/downloads/-/article/cbs-wijk-en-buurtkaart
```

Place the file in `data/raw/`, e.g.:

```
data/raw/wijkenbuurten_2023.gpkg
```

The relevant layer is typically `buurten_2023` (neighbourhoods) or `wijken_2023` (districts).

### 2. Klimaateffectatlas rasters

1. Go to [klimaateffectatlas.nl](https://www.klimaateffectatlas.nl)
2. Navigate to a theme (e.g. *Hitte > Gevoelstemperatuur*, *Wateroverlast > Overstroming*)
3. Download the GeoTIFF via the download button or WCS endpoint
4. Place rasters in `data/raw/`, e.g.:

```
data/raw/hitte_gevoelstemperatuur.tif
data/raw/droogte_neerslagtekort.tif
```

> All files in `data/raw/` are git-ignored to keep the repository lightweight.

---

## Quick Start & Testing

### 1. Configure `test_run.py`

Open `scripts/test_run.py` and fill in the file paths at the top of the file:

```python
# ✏️  FILL IN YOUR FILE PATHS HERE

CBS_GPKG       = Path("data/raw/wijkenbuurten_2023.gpkg")
CBS_LAYER      = "buurten_2023"
GEMEENTE       = "Eindhoven"                               # or Utrecht, Haarlem, Tilburg …
RASTER_HITTE   = Path("data/raw/hitte_gevoelstemperatuur.tif")
RASTER_DROOGTE = Path("data/raw/droogte_neerslagtekort.tif")  # set to None to skip
OUTPUT_PATH    = Path("output/test_eindhoven_klimaat.gpkg")
THRESHOLD      = 30.0   # pixels above this value → percentage column
```

### 2. Run the test

```bash
python scripts/test_run.py
```

Or with verbose logging:

```bash
python scripts/test_run.py --verbose
```

### 3. Expected console output

```
────────────────────────────────────────────────────
  klimaat-wijkprofielen-poc — test run
────────────────────────────────────────────────────
  Gemeente  : Eindhoven
  CBS laag  : buurten_2023
  Raster 1  : data/raw/hitte_gevoelstemperatuur.tif
  Raster 2  : data/raw/droogte_neerslagtekort.tif
  Drempel   : 30.0
  Output    : output/test_eindhoven_klimaat.gpkg

10:12:34  INFO     Laad wijken van wijkenbuurten_2023.gpkg (layer=buurten_2023)
10:12:34  INFO       3956 rijen geladen, CRS: EPSG:28992
10:12:34  INFO       Gefilterd op 'Eindhoven': 88 rijen over
10:12:35  INFO     Bereken zonal stats [mean, max, std, count] voor hitte_gevoelstemperatuur.tif
...

────────────────────────────────────────────────────
  Resultaten samenvatting
────────────────────────────────────────────────────
  Gemeente           : Eindhoven
  Aantal buurten     : 88
  Verwerkingstijd    : 4.3 seconden
  Uitvoerbestand     : output/test_eindhoven_klimaat.gpkg

  Zonal stat-kolommen (8):
    hitte_mean                           n= 88  min=   24.10  max=   33.80  gem=   29.41
    hitte_max                            n= 88  min=   28.50  max=   37.20  gem=   33.12
    hitte_std                            n= 88  min=    0.82  max=    3.14  gem=    1.73
    hitte_count                          n= 88  min=   12.00  max=  843.00  gem=  211.40
    droogte_mean                         n= 88  min=   55.20  max=  142.60  gem=   98.33
    ...

  Drempelwaarde-kolommen (>30.0) (2):
    hitte_pct_above_30.0                 n= 88  gemiddeld 43.2% boven drempel
    droogte_pct_above_30.0               n= 88  gemiddeld 91.7% boven drempel

  Genormaliseerde kolommen (2):
    hitte_mean_norm
    droogte_mean_norm

✅  Test run geslaagd!
```

### 4. What if files are missing?

The script checks for missing files before starting and prints clear instructions:

```
────────────────────────────────────────────────────
  ❌  Ontbrekende bestanden
────────────────────────────────────────────────────
  • data/raw/wijkenbuurten_2023.gpkg

  Vul de bestandspaden in bovenaan dit script (test_run.py),
  of raadpleeg README.md > "How to get the data" voor downloadinstructies.
```

---

## Usage

### Basic — single raster with auto-derived prefix

```bash
python scripts/enrich_wijken.py \
  --wijken   data/raw/wijkenbuurten_2023.gpkg \
  --layer    buurten_2023 \
  --rasters  data/raw/hitte_gevoelstemperatuur.tif \
  --output   output/buurten_hitte.gpkg
```

### Multiple rasters with explicit prefixes

```bash
python scripts/enrich_wijken.py \
  --wijken   data/raw/wijkenbuurten_2023.gpkg \
  --layer    buurten_2023 \
  --rasters  hitte=data/raw/hitte_gevoelstemperatuur.tif \
             droogte=data/raw/droogte_neerslagtekort.tif \
  --output   output/buurten_klimaat.gpkg
```

### Full example — filter by municipality, threshold + normalize

```bash
python scripts/enrich_wijken.py \
  --wijken     data/raw/wijkenbuurten_2023.gpkg \
  --layer      buurten_2023 \
  --rasters    hitte=data/raw/hitte_gevoelstemperatuur.tif \
               droogte=data/raw/droogte_neerslagtekort.tif \
  --gemeente   Eindhoven \
  --stats      mean max std count \
  --threshold  30 \
  --normalize \
  --output     output/eindhoven_klimaat.gpkg
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--wijken` | ✅ | Path to CBS GeoPackage or Shapefile |
| `--layer` | | Layer name inside the GeoPackage (default: first layer) |
| `--rasters` | ✅ | One or more rasters as `key=path.tif` or `path.tif` |
| `--gemeente` | | Filter on municipality name (GM_NAAM column) |
| `--stats` | | Statistics to compute (default: `mean max std count`) |
| `--threshold` | | Add a `{prefix}_pct_above_{threshold}` column per raster |
| `--normalize` | | Add min-max normalised `{prefix}_mean_norm` columns |
| `--output` | ✅ | Output GeoPackage path |
| `--verbose` / `-v` | | Enable DEBUG logging |

---

## Project structure

```
klimaat-wijkprofielen-poc/
├── .github/
│   └── workflows/              # CI/CD (future)
├── data/
│   ├── raw/                    # Source data — git-ignored
│   └── processed/              # Intermediate outputs
├── docs/                       # Documentation, methodology notes
├── output/                     # Final GeoPackages
├── scripts/
│   ├── enrich_wijken.py        # Main CLI script
│   ├── utils.py                # Shared helper functions
│   └── test_run.py             # Quick start / smoke test
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Next steps

- [ ] **Visualisatie** — add a choropleth export script using `matplotlib` / `contextily`
- [ ] **Automatische download** — fetch Klimaateffectatlas rasters via WCS in one command
- [ ] **Meerdere thema's** — extend with flood risk, urban greenery, and extreme rainfall layers
- [ ] **CI workflow** — add GitHub Actions to run a smoke test on synthetic data
- [ ] **Dashboard** — explore linking output to a lightweight web viewer (e.g. Felt, Kepler.gl)
- [ ] **Unit tests** — add pytest tests for `utils.py` functions

---

## License

This project is open source. Source data licences apply:
- CBS data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Klimaateffectatlas: see [gebruiksvoorwaarden](https://www.klimaateffectatlas.nl/nl/gebruiksvoorwaarden)