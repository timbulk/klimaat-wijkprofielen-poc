#!/usr/bin/env python3
"""
enrich_wijken.py
----------------
Enrich CBS Wijk- en Buurtenkaart polygons with climate impact data from the
Klimaateffectatlas by calculating zonal statistics per neighbourhood.

Rasters are passed as ``key=path`` pairs so the key becomes the column prefix:

    --rasters hitte=data/raw/hitte.tif droogte=data/raw/droogte.tif

When no explicit key is provided the prefix is derived from the filename stem.

Usage examples
--------------
# Single raster with auto-derived prefix
python scripts/enrich_wijken.py \
    --wijken  data/raw/wijkenbuurten_2023.gpkg \
    --rasters data/raw/hitte_gevoelstemperatuur.tif \
    --output  output/buurten_hitte.gpkg

# Multiple rasters with explicit prefixes, filtered to one municipality
python scripts/enrich_wijken.py \
    --wijken     data/raw/wijkenbuurten_2023.gpkg \
    --layer      buurten_2023 \
    --rasters    hitte=data/raw/hitte_gevoelstemperatuur.tif \
                 droogte=data/raw/droogte_neerslagtekort.tif \
    --gemeente   Amsterdam \
    --stats      mean max std count \
    --threshold  30 \
    --output     output/amsterdam_klimaat.gpkg
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import geopandas as gpd
from rasterstats import zonal_stats

# Local helpers — utils.py lives in the same directory
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
    gpkg_path:
        Path to the CBS GeoPackage (.gpkg) or Shapefile (.shp).
    layer:
        Layer name inside the GeoPackage.  When None the first layer is used.
    gemeente:
        Municipality name to filter on (GM_NAAM column).
        Pass None to keep all municipalities.

    Returns
    -------
    GeoDataFrame with the selected neighbourhood polygons.

    Raises
    ------
    FileNotFoundError
        When *gpkg_path* does not exist.
    ValueError
        When the GM_NAAM column is missing or *gemeente* yields no rows.
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
    gdf:
        GeoDataFrame aligned to the raster CRS — call
        :func:`utils.reproject_if_needed` before passing it here.
    raster_path:
        Path to a single-band GeoTIFF.
    stats:
        Statistics to compute, e.g. ``["mean", "max", "std", "count"]``.
    prefix:
        Column-name prefix.  Each stat becomes ``{prefix}_{stat}``.

    Returns
    -------
    GeoDataFrame with new columns appended (one per requested statistic).
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
        nodata=None,      # honour the raster's own nodata value
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
    gdf:
        Enriched GeoDataFrame to persist.
    output_path:
        Destination path.  The parent directory is created when absent.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Schrijf resultaat naar %s", output_path)
    gdf.to_file(output_path, driver="GPKG", engine="pyogrio")
    log.info("  Klaar — %d rijen, %d kolommen", len(gdf), len(gdf.columns))


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def derive_prefix(raster_path: Path) -> str:
    """Derive a short column prefix from the raster filename stem.

    Strips trailing year tags (``_2023``) and version tags (``_v2``) and
    truncates to 20 characters so column names stay manageable.

    Examples
    --------
    ``hitte_gevoelstemperatuur_2023.tif`` → ``hitte_gevoelstemperatuur``
    ``droogte_neerslagtekort.tif``        → ``droogte_neerslagtekort``
    """
    stem = raster_path.stem
    stem = re.sub(r"[_-]?(v\d+|\d{4})$", "", stem, flags=re.IGNORECASE)
    return stem[:20]


def parse_raster_args(raw: list[str]) -> dict[str, Path]:
    """Parse ``--rasters`` values into a ``{prefix: Path}`` mapping.

    Accepts two formats per entry:

    - ``key=path/to/file.tif``  — explicit prefix
    - ``path/to/file.tif``      — prefix derived from filename

    Parameters
    ----------
    raw:
        List of strings as received from argparse (``nargs="+"``).

    Returns
    -------
    Ordered dict mapping each prefix to its raster Path.

    Raises
    ------
    ValueError
        On duplicate prefixes (would cause column name collisions).
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

        if prefix in mapping:
            raise ValueError(
                f"Dubbel prefix '{prefix}' gedetecteerd. "
                "Gebruik expliciete sleutels: key=pad.tif."
            )
        mapping[prefix] = raster_path

    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enrich_wijken",
        description=(
            "Verrijk CBS wijken/buurten met klimaatdata uit de Klimaateffectatlas "
            "via zonal statistics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--wijken",
        type=Path,
        required=True,
        metavar="GPKG",
        help="Pad naar het CBS Wijk- en Buurtenkaart GeoPackage of Shapefile.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        metavar="NAAM",
        help="Laagnaam in het GeoPackage (standaard: eerste laag).",
    )
    parser.add_argument(
        "--rasters",
        nargs="+",
        required=True,
        metavar="[KEY=]PAD",
        help=(
            "Een of meer rasters als 'key=pad.tif' (expliciete prefix) of "
            "'pad.tif' (prefix afgeleid van bestandsnaam). "
            "Voorbeeld: hitte=data/raw/hitte.tif droogte=data/raw/droogte.tif"
        ),
    )
    parser.add_argument(
        "--gemeente",
        type=str,
        default=None,
        metavar="NAAM",
        help="Filter op gemeentenaam (GM_NAAM kolom), bijv. 'Amsterdam'.",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        default=["mean", "max", "std", "count"],
        metavar="STAT",
        choices=AVAILABLE_STATS,
        help=(
            f"Te berekenen statistieken (standaard: mean max std count). "
            f"Keuze uit: {', '.join(AVAILABLE_STATS)}."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="WAARDE",
        help=(
            "Bereken ook het percentage pixels boven deze drempelwaarde per raster. "
            "Voegt een kolom '{prefix}_pct_above_{threshold}' toe."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Voeg genormaliseerde versies (0-1) toe van alle mean-kolommen. "
            "Handig voor het maken van een samengestelde risicoscore."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="GPKG",
        help="Pad voor het uitvoer GeoPackage, bijv. output/buurten_klimaat.gpkg.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Toon uitgebreide debug-informatie.",
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
        # 1. Parse raster arguments → {prefix: Path}
        raster_map = parse_raster_args(args.rasters)
        log.info("Rasters: %s", {k: v.name for k, v in raster_map.items()})

        # 2. Load CBS polygons
        gdf = load_wijken(args.wijken, args.layer, args.gemeente)

        # 3. Process each raster
        for prefix, raster_path in raster_map.items():
            # Align CRS using the util helper
            gdf = reproject_if_needed(gdf, raster_path)

            # Standard zonal statistics
            gdf = compute_zonal_stats(gdf, raster_path, args.stats, prefix)

            # Optional threshold percentage
            if args.threshold is not None:
                log.info(
                    "Bereken percentage boven drempel %.2f voor '%s'",
                    args.threshold,
                    prefix,
                )
                gdf = calculate_percentage_above_threshold(
                    gdf, raster_path, args.threshold, prefix
                )

            # Optional min-max normalisation of the mean column
            if args.normalize and f"{prefix}_mean" in gdf.columns:
                col = f"{prefix}_mean"
                gdf[f"{col}_norm"] = normalize_column(gdf, col)
                log.info("  Genormaliseerde kolom toegevoegd: %s_norm", col)

        # 4. Save result
        save_output(gdf, args.output)

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