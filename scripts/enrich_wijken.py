#!/usr/bin/env python3
"""
enrich_wijken.py
----------------
Enrich CBS Wijk- en Buurtenkaart polygons with climate impact data via
zonal statistics.  Raster data can come from:

  1. Local GeoTIFF files  (--rasters key=path.tif …)
  2. A WMS endpoint       (--wms-layer hitteeiland_r_hitte)

Both sources can be combined in a single run.  Config defaults are read from
config.yaml; every setting can be overridden on the CLI.

Usage examples
--------------
# Use config.yaml for everything (local rasters)
python scripts/enrich_wijken.py

# Override gemeente; rest from config
python scripts/enrich_wijken.py --gemeente Utrecht

# WMS only — no local rasters needed
python scripts/enrich_wijken.py \
    --wijken    data/raw/wijkenbuurten_2023.gpkg \
    --gemeente  Eindhoven \
    --wms-layer hitteeiland_r_hitte \
    --output    output/eindhoven_hitte_wms.gpkg

# Local rasters + WMS in one run
python scripts/enrich_wijken.py \
    --rasters   droogte=data/raw/droogte.tif \
    --wms-layer hitteeiland_r_hitte \
    --gemeente  Utrecht \
    --threshold 30
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml
from rasterstats import zonal_stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    calculate_percentage_above_threshold,
    normalize_column,
    reproject_if_needed,
)
from wms_utils import (
    DEFAULT_WMS_URL,
    DEFAULT_WMS_LAYER,
    DEFAULT_RESOLUTION_M,
    DEFAULT_BUFFER_M,
    TempRaster,
    download_wms_as_geotiff,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"

# ---------------------------------------------------------------------------
# Config loading & resolution
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config; return empty dict when the file is absent."""
    if not config_path.exists():
        log.debug("Geen config op %s — alleen CLI gebruikt", config_path)
        return {}
    log.info("Laad configuratie van %s", config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config-file defaults with CLI arguments.

    CLI values always win.  None in *args* means "not supplied on CLI".

    Returns a resolved settings dict with keys:
    wijken, layer, rasters, wms_layer, wms_url, gemeente, stats,
    threshold, normalize, output.
    """
    root = Path(__file__).parent.parent

    def _abs(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    # ── CBS input ────────────────────────────────────────────────────────────
    wijken_raw = args.wijken or cfg.get("cbs_path")
    if not wijken_raw:
        raise ValueError("Geen CBS-bestand opgegeven (--wijken of cbs_path in config).")
    wijken = _abs(wijken_raw)
    layer = args.layer if args.layer is not None else cfg.get("cbs_layer")

    # ── Local rasters ────────────────────────────────────────────────────────
    raster_map: dict[str, Path] = {}
    if args.rasters:
        raster_map = parse_raster_args(args.rasters, root)
    elif cfg.get("rasters"):
        for key, path_str in cfg["rasters"].items():
            raster_map[key] = _abs(path_str)

    # ── WMS settings ─────────────────────────────────────────────────────────
    # --wms-layer "" (empty string) disables WMS even when a default exists.
    wms_layer: str | None
    if args.wms_layer is not None:
        wms_layer = args.wms_layer if args.wms_layer else None
    else:
        wms_layer = cfg.get("wms_layer")  # may be absent → None

    wms_url = cfg.get("wms_url", DEFAULT_WMS_URL)

    # At least one raster source must be present
    if not raster_map and not wms_layer:
        raise ValueError(
            "Geen rasters opgegeven.  Gebruik --rasters, --wms-layer, "
            "of definieer rasters/wms_layer in config.yaml."
        )

    # ── Other settings ───────────────────────────────────────────────────────
    gemeente  = args.gemeente if args.gemeente is not None else cfg.get("gemeente")
    stats     = args.stats or cfg.get("stats") or ["mean", "max", "std", "count"]
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold")
    normalize = args.normalize

    # ── Output path ──────────────────────────────────────────────────────────
    if args.output:
        output = args.output
    else:
        out_dir = root / (cfg.get("output_dir") or "output")
        slug = (gemeente or "all").lower().replace(" ", "_")
        output = out_dir / f"{slug}_klimaat.gpkg"

    return {
        "wijken":    wijken,
        "layer":     layer,
        "rasters":   raster_map,
        "wms_layer": wms_layer,
        "wms_url":   wms_url,
        "gemeente":  gemeente,
        "stats":     stats,
        "threshold": threshold,
        "normalize": normalize,
        "output":    output,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Candidate column names for the municipality name, in order of preference.
# CBS changed the column name between dataset editions:
#   - Older editions (up to ~2022):  GM_NAAM
#   - Newer editions (2023+):        gemeentenaam
_GEMEENTE_COLS = ("GM_NAAM", "gemeentenaam")


def _find_gemeente_col(gdf: gpd.GeoDataFrame) -> str:
    """Return the municipality-name column present in *gdf*.

    Tries each name in ``_GEMEENTE_COLS`` in order.  Raises a clear
    :class:`ValueError` listing all available columns when none is found,
    so the user knows exactly what to look for.

    Parameters
    ----------
    gdf: Loaded CBS GeoDataFrame.

    Raises
    ------
    ValueError  When no known municipality-name column is present.
    """
    for candidate in _GEMEENTE_COLS:
        if candidate in gdf.columns:
            log.debug("Gemeente-kolom gevonden: '%s'", candidate)
            return candidate

    raise ValueError(
        f"Geen gemeente-naamkolom gevonden. "
        f"Gezocht naar: {list(_GEMEENTE_COLS)}. "
        f"Beschikbare kolommen: {list(gdf.columns)}. "
        "Controleer de CBS-laagnaam of pas _GEMEENTE_COLS aan in enrich_wijken.py."
    )


def load_wijken(
    gpkg_path: Path,
    layer: str | None,
    gemeente: str | None,
) -> gpd.GeoDataFrame:
    """Load CBS neighbourhoods and optionally filter by municipality name.

    The CBS dataset has used different column names for the municipality name
    across editions (``GM_NAAM`` in older files, ``gemeentenaam`` from 2023+).
    This function auto-detects the correct column via :func:`_find_gemeente_col`.

    Parameters
    ----------
    gpkg_path:  Path to CBS GeoPackage or Shapefile.
    layer:      Layer name inside the GeoPackage (None → first layer).
    gemeente:   Municipality name to filter on, case-insensitive.
                None → return all municipalities unchanged.

    Returns
    -------
    Filtered GeoDataFrame with the selected neighbourhood polygons.

    Raises
    ------
    FileNotFoundError  When *gpkg_path* does not exist.
    ValueError         When no municipality-name column is found, or when
                       *gemeente* matches no rows.
    """
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Wijkenbestand niet gevonden: {gpkg_path}")

    log.info("Laad wijken van %s (layer=%s)", gpkg_path.name, layer or "eerste laag")
    # Pass layer=0 (first layer) when no layer name is given, as required by pyogrio/fiona
    gdf = gpd.read_file(gpkg_path, layer=layer if layer else 0, engine="pyogrio")
    log.info("  %d rijen geladen, CRS: %s", len(gdf), gdf.crs)

    if gemeente:
        # Auto-detect the municipality name column (GM_NAAM or gemeentenaam)
        col = _find_gemeente_col(gdf)
        log.info("  Filter op kolom '%s' = '%s'", col, gemeente)

        # Exact match — CBS values already have correct casing (e.g. "Eindhoven")
        gdf = gdf[gdf[col] == gemeente].copy()

        if len(gdf) == 0:
            # Provide a sample of real values so the user can spot typos
            sample = sorted(gdf[col].dropna().unique()[:10].tolist()) if col in gdf.columns else []
            raise ValueError(
                f"Geen rijen gevonden voor gemeente '{gemeente}' in kolom '{col}'. "
                f"Controleer de schrijfwijze (hoofdlettergevoelig). "
                + (f"Voorbeeldwaarden: {sample}" if sample else "")
            )

        log.info("  Gefilterd op '%s': %d rijen", gemeente, len(gdf))

    return gdf


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

AVAILABLE_STATS = [
    "mean", "max", "min", "std", "count", "sum",
    "median", "range", "majority", "minority",
    "variety", "percentile_25", "percentile_75",
]


def compute_zonal_stats(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    stats: list[str],
    prefix: str,
) -> gpd.GeoDataFrame:
    """Calculate zonal statistics for every polygon against one raster.

    Parameters
    ----------
    gdf:          GeoDataFrame in the same CRS as *raster_path*.
    raster_path:  Path to a single-band GeoTIFF (local or temp WMS file).
    stats:        Statistics to compute.
    prefix:       Column-name prefix → ``{prefix}_{stat}`` per stat.
    """
    log.info(
        "Bereken zonal stats [%s] voor %s → prefix '%s'",
        ", ".join(stats), raster_path.name, prefix,
    )

    # Read nodata value from the raster file so pixels tagged as nodata
    # (e.g. water / outside study area = 0 in the KEA hitteeiland layer)
    # are excluded from all statistics instead of being counted as real values.
    import rasterio as _rio
    with _rio.open(str(raster_path)) as _src:
        _nodata = _src.nodata  # e.g. 0.0 for hitteeiland
    results = zonal_stats(gdf, str(raster_path), stats=stats,
                          nodata=_nodata, all_touched=False)
    gdf = gdf.copy()
    for stat in stats:
        col = f"{prefix}_{stat}"
        gdf[col] = [row.get(stat) for row in results]
        log.debug("  Kolom: %s", col)

    log.info("  %d kolommen toegevoegd", len(stats))
    return gdf


def _enrich_from_raster(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    prefix: str,
    stats: list[str],
    threshold: float | None,
    normalize: bool,
) -> gpd.GeoDataFrame:
    """Run the full per-raster enrichment pipeline (reproject → stats → threshold → normalise).

    Extracted as a helper so both local-file and WMS paths share the same logic.
    """
    gdf = reproject_if_needed(gdf, raster_path)
    gdf = compute_zonal_stats(gdf, raster_path, stats, prefix)

    if threshold is not None:
        log.info("Drempelwaarde %.2f voor '%s'", threshold, prefix)
        gdf = calculate_percentage_above_threshold(gdf, raster_path, threshold, prefix)

    if normalize and f"{prefix}_mean" in gdf.columns:
        col = f"{prefix}_mean"
        gdf[f"{col}_norm"] = normalize_column(gdf, col)
        log.info("  Genormaliseerde kolom: %s_norm", col)

    return gdf


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_output(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write the enriched GeoDataFrame to a GeoPackage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Schrijf resultaat naar %s", output_path)
    gdf.to_file(output_path, driver="GPKG", engine="pyogrio")
    log.info("  Klaar — %d rijen, %d kolommen", len(gdf), len(gdf.columns))


# ---------------------------------------------------------------------------
# Argument-parsing helpers
# ---------------------------------------------------------------------------

def derive_prefix(raster_path: Path) -> str:
    """Derive a short (<= 20 char) column prefix from the raster filename stem."""
    stem = raster_path.stem
    stem = re.sub(r"[_-]?(v\d+|\d{4})$", "", stem, flags=re.IGNORECASE)
    return stem[:20]


def parse_raster_args(raw: list[str], root: Path | None = None) -> dict[str, Path]:
    """Parse ``key=path`` or plain ``path`` entries into ``{prefix: Path}``.

    Raises
    ------
    ValueError  On duplicate prefixes.
    """
    mapping: dict[str, Path] = {}
    for entry in raw:
        if "=" in entry:
            key, _, path_str = entry.partition("=")
            prefix = key.strip()
            rp = Path(path_str.strip())
        else:
            rp = Path(entry.strip())
            prefix = derive_prefix(rp)

        if root and not rp.is_absolute():
            rp = root / rp

        if prefix in mapping:
            raise ValueError(f"Dubbel prefix '{prefix}'. Gebruik expliciete sleutels: key=pad.tif.")
        mapping[prefix] = rp
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enrich_wijken",
        description=(
            "Verrijk CBS wijken/buurten met klimaatdata via zonal statistics. "
            "Leest standaard uit config.yaml; CLI-argumenten hebben voorrang."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, metavar="YAML",
        help=f"YAML-configuratiebestand (standaard: {DEFAULT_CONFIG.name}).")

    # CBS input
    parser.add_argument("--wijken", type=Path, default=None, metavar="GPKG",
        help="CBS GeoPackage of Shapefile (overschrijft cbs_path in config).")
    parser.add_argument("--layer", type=str, default=None, metavar="NAAM",
        help="Laagnaam in het GeoPackage.")

    # Local rasters
    parser.add_argument("--rasters", nargs="+", default=None, metavar="[KEY=]PAD",
        help="Lokale rasters als 'key=pad.tif' of 'pad.tif'. Overschrijft config.")

    # WMS raster source
    wms_group = parser.add_argument_group(
        "WMS",
        "Download een rasterlaag rechtstreeks van een WMS-endpoint.\n"
        "Kan worden gecombineerd met --rasters voor een gemengde run.",
    )
    wms_group.add_argument(
        "--wms-layer",
        type=str,
        default=None,
        metavar="LAAG",
        dest="wms_layer",
        help=(
            f"WMS-laagnaam om te downloaden als tijdelijk GeoTIFF. "
            f"Standaard WMS-laag: '{DEFAULT_WMS_LAYER}'. "
            "Geef een lege string ('') om WMS uit te schakelen ook al staat het in config."
        ),
    )
    wms_group.add_argument(
        "--wms-url",
        type=str,
        default=None,
        metavar="URL",
        dest="wms_url",
        help=f"WMS-endpoint URL (standaard: Sogelink KEA public WMS).",
    )
    wms_group.add_argument(
        "--wms-resolution",
        type=int,
        default=50,
        metavar="METER",
        dest="wms_resolution",
        help="Pixelgrootte in meters voor WMS-download (standaard: 50).",
    )
    wms_group.add_argument(
        "--wms-buffer",
        type=int,
        default=500,
        metavar="METER",
        dest="wms_buffer",
        help="Buffer in meters rondom de gemeente-bbox (standaard: 500).",
    )

    # Filter & stats
    parser.add_argument("--gemeente", type=str, default=None, metavar="NAAM",
        help="Filter op gemeentenaam (GM_NAAM).")
    parser.add_argument("--stats", nargs="+", default=None, metavar="STAT",
        choices=AVAILABLE_STATS,
        help="Te berekenen statistieken (standaard: mean max std count).")
    parser.add_argument("--threshold", type=float, default=None, metavar="WAARDE",
        help="Drempelwaarde voor percentage-boven-kolom.")
    parser.add_argument("--normalize", action="store_true",
        help="Voeg genormaliseerde (0-1) mean-kolommen toe.")

    # Output
    parser.add_argument("--output", type=Path, default=None, metavar="GPKG",
        help="Uitvoerpad (standaard: {output_dir}/{gemeente}_klimaat.gpkg).")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Toon DEBUG-logberichten.")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cfg = load_config(args.config)

        # Allow --wms-url CLI to override config
        if args.wms_url:
            cfg["wms_url"] = args.wms_url

        settings = resolve_config(cfg, args)

        log.info(
            "Instellingen: gemeente=%s  rasters=%s  wms_layer=%s  stats=%s  threshold=%s",
            settings["gemeente"] or "alle",
            list(settings["rasters"].keys()) or "—",
            settings["wms_layer"] or "—",
            settings["stats"],
            settings["threshold"],
        )

        # 1. Load CBS polygons
        gdf = load_wijken(settings["wijken"], settings["layer"], settings["gemeente"])

        # 2a. Process local rasters
        for prefix, raster_path in settings["rasters"].items():
            gdf = _enrich_from_raster(
                gdf, raster_path, prefix,
                settings["stats"], settings["threshold"], settings["normalize"],
            )

        # 2b. Process WMS layer (downloads to a temp file, deleted after use)
        if settings["wms_layer"]:
            wms_prefix = settings["wms_layer"].replace("_r_", "_").replace("_", "_")[:20]
            log.info("WMS-laag '%s' → prefix '%s'", settings["wms_layer"], wms_prefix)

            with TempRaster(suffix=f"_{settings['wms_layer']}.tif") as tmp_path:
                try:
                    download_wms_as_geotiff(
                        gdf,
                        wms_url=settings["wms_url"],
                        layer_name=settings["wms_layer"],
                        resolution_m=args.wms_resolution,
                        buffer_m=args.wms_buffer,
                        output_path=tmp_path,
                    )
                    gdf = _enrich_from_raster(
                        gdf, tmp_path, wms_prefix,
                        settings["stats"], settings["threshold"], settings["normalize"],
                    )
                except (ConnectionError, ValueError, RuntimeError) as exc:
                    log.error("WMS-verwerking mislukt: %s", exc)
                    log.error("Zorg dat de WMS bereikbaar is en de laagnaam klopt.")
                    return 1

        # 3. Save
        save_output(gdf, settings["output"])

    except FileNotFoundError as exc:
        log.error("Bestand niet gevonden: %s", exc)
        return 1
    except ValueError as exc:
        log.error("Ongeldige invoer: %s", exc)
        return 1
    except Exception as exc:
        log.exception("Onverwachte fout: %s", exc)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())