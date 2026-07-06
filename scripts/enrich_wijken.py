#!/usr/bin/env python3
"""
enrich_wijken.py
----------------
Enrich CBS Wijk- en Buurtenkaart polygons with climate impact data from the
Klimaateffectatlas by calculating zonal statistics per neighbourhood.

Configuration is read from config.yaml (project root) and can be fully
overridden via command-line arguments.  CLI flags always win over config.

Rasters can be passed as ``key=path`` pairs so the key becomes the column
prefix, or as plain paths (prefix derived from filename stem):

    --rasters hitte=data/raw/hitte.tif droogte=data/raw/droogte.tif

Usage examples
--------------
# Use config.yaml for everything
python scripts/enrich_wijken.py

# Use a different config file
python scripts/enrich_wijken.py --config config.local.yaml

# Override gemeente and output on the CLI
python scripts/enrich_wijken.py --gemeente Utrecht --output output/utrecht.gpkg

# Single raster run, no config file
python scripts/enrich_wijken.py \
    --wijken  data/raw/wijkenbuurten_2023.gpkg \
    --rasters hitte=data/raw/hitte.tif \
    --output  output/buurten_hitte.gpkg
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Default config file relative to the project root
DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load and return the YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.

    Returns
    -------
    Parsed config as a plain dict.  Returns an empty dict when the file
    does not exist so callers can always treat the result as a dict.
    """
    if not config_path.exists():
        log.debug("Geen config gevonden op %s — alleen CLI-argumenten gebruikt", config_path)
        return {}

    log.info("Laad configuratie van %s", config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    return cfg


def resolve_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config-file values with CLI arguments.

    CLI arguments always take precedence.  A value of None in *args* means
    "not supplied on the CLI" and the config-file default is used instead.

    Parameters
    ----------
    cfg:   Parsed config.yaml dict (may be empty).
    args:  Parsed argparse Namespace.

    Returns
    -------
    Resolved settings dict with keys: wijken, layer, rasters, gemeente,
    stats, threshold, normalize, output.
    """
    root = Path(__file__).parent.parent  # project root for relative paths

    def _path(val: str | None) -> Path | None:
        return (root / val) if val else None

    # ── wijken ──────────────────────────────────────────────────────────────
    wijken_raw = args.wijken or cfg.get("cbs_path")
    if not wijken_raw:
        raise ValueError("Geen CBS-bestand opgegeven (--wijken of cbs_path in config).")
    wijken = Path(wijken_raw) if Path(wijken_raw).is_absolute() else root / wijken_raw

    # ── layer ───────────────────────────────────────────────────────────────
    layer = args.layer if args.layer is not None else cfg.get("cbs_layer")

    # ── rasters ─────────────────────────────────────────────────────────────
    raster_map: dict[str, Path] = {}
    if args.rasters:
        raster_map = parse_raster_args(args.rasters, root)
    elif cfg.get("rasters"):
        for key, path_str in cfg["rasters"].items():
            raster_map[key] = root / path_str if not Path(path_str).is_absolute() else Path(path_str)
    if not raster_map:
        raise ValueError("Geen rasters opgegeven (--rasters of rasters in config).")

    # ── gemeente ─────────────────────────────────────────────────────────────
    gemeente = args.gemeente if args.gemeente is not None else cfg.get("gemeente")

    # ── stats ────────────────────────────────────────────────────────────────
    stats = args.stats if args.stats else (cfg.get("stats") or ["mean", "max", "std", "count"])

    # ── threshold ────────────────────────────────────────────────────────────
    threshold: float | None
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = cfg.get("threshold")  # may be None when set to null in yaml

    # ── normalize ────────────────────────────────────────────────────────────
    normalize = args.normalize  # boolean flag — False when not on CLI; no config equiv.

    # ── output ───────────────────────────────────────────────────────────────
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
        "gemeente":  gemeente,
        "stats":     stats,
        "threshold": threshold,
        "normalize": normalize,
        "output":    output,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wijken(
    gpkg_path: Path,
    layer: str | None,
    gemeente: str | None,
) -> gpd.GeoDataFrame:
    """Load CBS neighbourhoods from a GeoPackage and optionally filter by municipality.

    Parameters
    ----------
    gpkg_path:  Path to the CBS GeoPackage (.gpkg) or Shapefile (.shp).
    layer:      Layer name inside the GeoPackage. None → first layer.
    gemeente:   Municipality name to filter on (GM_NAAM). None → keep all.

    Returns
    -------
    GeoDataFrame with the selected neighbourhood polygons.

    Raises
    ------
    FileNotFoundError  When *gpkg_path* does not exist.
    ValueError         When GM_NAAM is missing or *gemeente* yields no rows.
    """
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Wijkenbestand niet gevonden: {gpkg_path}")

    log.info("Laad wijken van %s (layer=%s)", gpkg_path.name, layer or "eerste laag")
    gdf = gpd.read_file(gpkg_path, layer=layer, engine="pyogrio")
    log.info("  %d rijen geladen, CRS: %s", len(gdf), gdf.crs)

    if gemeente:
        col = "GM_NAAM"
        if col not in gdf.columns:
            raise ValueError(
                f"Kolom '{col}' niet gevonden. "
                f"Beschikbare kolommen: {list(gdf.columns)}"
            )
        mask = gdf[col].str.strip().str.lower() == gemeente.strip().lower()
        gdf = gdf[mask].copy()
        if gdf.empty:
            raise ValueError(
                f"Geen rijen gevonden voor gemeente '{gemeente}'. "
                "Controleer de schrijfwijze."
            )
        log.info("  Gefilterd op '%s': %d rijen over", gemeente, len(gdf))

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
    gdf:          GeoDataFrame aligned to the raster CRS.
    raster_path:  Path to a single-band GeoTIFF.
    stats:        Statistics to compute, e.g. ``["mean", "max", "std", "count"]``.
    prefix:       Column-name prefix → ``{prefix}_{stat}`` per statistic.

    Returns
    -------
    GeoDataFrame with new stat columns appended.
    """
    log.info(
        "Bereken zonal stats [%s] voor %s → prefix '%s'",
        ", ".join(stats),
        raster_path.name,
        prefix,
    )

    results = zonal_stats(
        gdf,
        str(raster_path),
        stats=stats,
        nodata=None,
        all_touched=False,
    )

    gdf = gdf.copy()
    for stat in stats:
        col = f"{prefix}_{stat}"
        gdf[col] = [row.get(stat) for row in results]
        log.debug("  Kolom toegevoegd: %s", col)

    log.info("  %d statistiekkolommen toegevoegd", len(stats))
    return gdf


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_output(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write the enriched GeoDataFrame to a GeoPackage.

    Parameters
    ----------
    gdf:          Enriched GeoDataFrame.
    output_path:  Destination path; parent directory is created when absent.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Schrijf resultaat naar %s", output_path)
    gdf.to_file(output_path, driver="GPKG", engine="pyogrio")
    log.info("  Klaar — %d rijen, %d kolommen", len(gdf), len(gdf.columns))


# ---------------------------------------------------------------------------
# Argument-parsing helpers
# ---------------------------------------------------------------------------

def derive_prefix(raster_path: Path) -> str:
    """Derive a short column prefix from the raster filename stem.

    Strips trailing year/version tags and truncates to 20 characters.

    Examples
    --------
    ``hitte_gevoelstemperatuur_2023.tif`` → ``hitte_gevoelstemperatuur``
    ``droogte_neerslagtekort.tif``        → ``droogte_neerslagtekort``
    """
    stem = raster_path.stem
    stem = re.sub(r"[_-]?(v\d+|\d{4})$", "", stem, flags=re.IGNORECASE)
    return stem[:20]


def parse_raster_args(raw: list[str], root: Path | None = None) -> dict[str, Path]:
    """Parse ``--rasters`` values into a ``{prefix: Path}`` mapping.

    Accepts two formats per entry:

    - ``key=path/to/file.tif``  — explicit prefix
    - ``path/to/file.tif``      — prefix derived from filename

    Parameters
    ----------
    raw:   List of strings from argparse (``nargs="+"``).
    root:  Optional project root; relative paths are resolved against it.

    Returns
    -------
    Ordered dict mapping each prefix to its resolved raster Path.

    Raises
    ------
    ValueError  On duplicate prefixes.
    """
    mapping: dict[str, Path] = {}
    for entry in raw:
        if "=" in entry:
            key, _, path_str = entry.partition("=")
            prefix = key.strip()
            raster_path = Path(path_str.strip())
        else:
            raster_path = Path(entry.strip())
            prefix = derive_prefix(raster_path)

        if root and not raster_path.is_absolute():
            raster_path = root / raster_path

        if prefix in mapping:
            raise ValueError(
                f"Dubbel prefix '{prefix}'. "
                "Gebruik expliciete sleutels: key=pad.tif."
            )
        mapping[prefix] = raster_path

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

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        metavar="YAML",
        help=f"Pad naar het YAML-configuratiebestand (standaard: {DEFAULT_CONFIG.name}).",
    )
    parser.add_argument(
        "--wijken",
        type=Path,
        default=None,
        metavar="GPKG",
        help="Pad naar het CBS GeoPackage of Shapefile (overschrijft cbs_path in config).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        metavar="NAAM",
        help="Laagnaam in het GeoPackage (overschrijft cbs_layer in config).",
    )
    parser.add_argument(
        "--rasters",
        nargs="+",
        default=None,
        metavar="[KEY=]PAD",
        help=(
            "Rasters als 'key=pad.tif' of 'pad.tif'. "
            "Overschrijft rasters in config."
        ),
    )
    parser.add_argument(
        "--gemeente",
        type=str,
        default=None,
        metavar="NAAM",
        help="Filter op gemeentenaam (overschrijft gemeente in config).",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        default=None,
        metavar="STAT",
        choices=AVAILABLE_STATS,
        help="Te berekenen statistieken (overschrijft stats in config).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="WAARDE",
        help="Drempelwaarde voor percentage-boven-kolom (overschrijft threshold in config).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Voeg genormaliseerde (0-1) mean-kolommen toe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="GPKG",
        help="Uitvoerpad (standaard: {output_dir}/{gemeente}_klimaat.gpkg).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Toon DEBUG-logberichten.",
    )

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
        # 1. Load config and merge with CLI args
        cfg = load_config(args.config)
        settings = resolve_config(cfg, args)

        log.info(
            "Instellingen: gemeente=%s  rasters=%s  stats=%s  threshold=%s",
            settings["gemeente"] or "alle",
            list(settings["rasters"].keys()),
            settings["stats"],
            settings["threshold"],
        )

        # 2. Load CBS polygons
        gdf = load_wijken(settings["wijken"], settings["layer"], settings["gemeente"])

        # 3. Process each raster
        for prefix, raster_path in settings["rasters"].items():
            gdf = reproject_if_needed(gdf, raster_path)
            gdf = compute_zonal_stats(gdf, raster_path, settings["stats"], prefix)

            if settings["threshold"] is not None:
                log.info("Drempelwaarde %.2f voor '%s'", settings["threshold"], prefix)
                gdf = calculate_percentage_above_threshold(
                    gdf, raster_path, settings["threshold"], prefix
                )

            if settings["normalize"] and f"{prefix}_mean" in gdf.columns:
                col = f"{prefix}_mean"
                gdf[f"{col}_norm"] = normalize_column(gdf, col)
                log.info("  Genormaliseerde kolom: %s_norm", col)

        # 4. Save result
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