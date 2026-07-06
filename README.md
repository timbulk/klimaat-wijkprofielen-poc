# klimaat-wijkprofielen-poc

> **Proof of concept** — Enrich CBS neighbourhood polygons with climate impact raster data from the Klimaateffectatlas using zonal statistics.

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

## Usage

### Basic — single raster

```bash
python scripts/bereken_wijkprofielen.py \
  --wijken   data/raw/wijkenbuurten_2023.gpkg \
  --layer    buurten_2023 \
  --raster   data/raw/hitte_gevoelstemperatuur.tif \
  --stats    mean max \
  --output   output/buurten_hitte.gpkg
```

### Multiple rasters in one run

```bash
python scripts/bereken_wijkprofielen.py \
  --wijken   data/raw/wijkenbuurten_2023.gpkg \
  --layer    buurten_2023 \
  --raster   data/raw/hitte_gevoelstemperatuur.tif \
             data/raw/droogte_neerslagtekort.tif \
  --stats    mean max std \
  --output   output/buurten_klimaat.gpkg
```

### Filter to a single municipality

```bash
python scripts/bereken_wijkprofielen.py \
  --wijken      data/raw/wijkenbuurten_2023.gpkg \
  --layer       buurten_2023 \
  --raster      data/raw/hitte_gevoelstemperatuur.tif \
  --filter-col  GM_NAAM \
  --filter-val  "Amsterdam" \
  --stats       mean max \
  --output      output/amsterdam_hitte.gpkg
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--wijken` | ✅ | Path to CBS GeoPackage or Shapefile |
| `--layer` | | Layer name inside the GeoPackage (default: first layer) |
| `--raster` | ✅ | One or more GeoTIFF paths |
| `--stats` | | Statistics to compute (default: `mean max`) |
| `--filter-col` | | Column name to filter on (e.g. `GM_NAAM`) |
| `--filter-val` | | Value to filter on (e.g. `Amsterdam`) |
| `--output` | ✅ | Output GeoPackage path |

---

## Project structure

```
klimaat-wijkprofielen-poc/
├── .github/
│   └── workflows/          # CI/CD (future)
├── data/
│   ├── raw/                # Source data — git-ignored
│   └── processed/          # Intermediate outputs
├── docs/                   # Documentation, methodology notes
├── output/                 # Final GeoPackages
├── scripts/
│   └── bereken_wijkprofielen.py   # Main analysis script
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Next steps

- [ ] **Script implementatie** — write `scripts/bereken_wijkprofielen.py` with full CLI
- [ ] **Meerdere thema's** — extend with flood risk, drought, and urban greenery layers
- [ ] **Visualisatie** — add a choropleth export script using `matplotlib` / `contextily`
- [ ] **Automatische download** — fetch Klimaateffectatlas rasters via WCS in one command
- [ ] **CI workflow** — add GitHub Actions to run a smoke test on synthetic data
- [ ] **Dashboard** — explore linking output to a lightweight web viewer (e.g. Felt, Kepler.gl)

---

## License

This project is open source. Source data licences apply:
- CBS data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Klimaateffectatlas: see [gebruiksvoorwaarden](https://www.klimaateffectatlas.nl/nl/gebruiksvoorwaarden)