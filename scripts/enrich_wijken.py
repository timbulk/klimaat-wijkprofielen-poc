#!/usr/bin/env python3
"""
enrich_wijken.py
----------------
Enrich CBS Wijk- en Buurtenkaart polygons with climate impact data from the
Klimaateffectatlas by calculating zonal statistics (mean, max, std, count)
per neighbourhood or district.

Usage examples
--------------
# Single raster, all municipalities
python scripts/enrich_wijken.py \
    --wijken   data/raw/wijkenbuurten_2023.gpkg \
    --layer    buurten_2023 \
    --raster   data/raw/hitte_gevoelstemperatuur.tif \
    --output   output/buurten_hitte.gpkg

# Multiple rasters, filtered to one municipality
python scripts/enrich_wijken.py \
    --wijken      data/raw/wijkenbuurten_2023.gpkg \
    --layer       buurten_2023 \
    --raster      data/raw/hitte_gevoelstemperatuur.tif data/raw/droogte_neerslagtekort.tif \
    --gemeente    Amsterdam \
    --stats       mean max std count \
    --output      output/amsterdam_klimaat.gpkg
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

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

def load_wijken(gpkg_path: Path, layer: str | None, gemeente: str | None) -> gpd.GeoDataFrame:
    """Load CBS neighbourhoods from a GeoPackage and optionally filter by municipality.

    Parameters
    ----------
    gpkg_path:  Path to the CBS GeoPackage (.gpkg) or Shapefile (.shp).
    layer:      Layer name inside the GeoPackage. If None the first layer is used.
    gemeente:   Municipality name to filter on (GM_NAAM column).
                Pass None to keep all municipalities.

    Returns
    -------
    GeoDataFrame with the selected neighbourhood polygons.
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
                f"Kolom '{col}' niet gevonden. Beschikbare kolommen: {list(gdf.columns)}"
            )
        gdf = gdf[gdf[col].str.strip().str.lower() == gemeente.strip().lower()].copy()
        if gdf.empty:
            raise ValueError(
                f"Geen rijen gevonden voor gemeente '{gemeente}'. "
                f"Controleer de schrijfwijze (hoofdlettergevoelig)."
            )
        log.info("  Gefilterd op '%s': %d rijen over", gemeente, len(gdf))

    return gdf


# ---------------------------------------------------------------------------
# CRS alignment
# ---------------------------------------------------------------------------

def align_crs(gdf: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    """Reproject *gdf* to the CRS of *raster_path* when they differ.

    rasterstats requires vector and raster to share the same CRS.

    Parameters
    ----------
    gdf:          Input GeoDataFrame.
    raster_path:  Path to the reference raster.

    Returns
    -------
    GeoDataFrame in the raster CRS (may be the original object if CRS matched).
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if gdf.crs is None:
        log.warning("Vector heeft geen CRS — neem raster-CRS aan (%s)", raster_crs)
        gdf = gdf.set_crs(raster_crs)
    elif gdf.crs != raster_crs:
        log.info(
            "Reprojecteer vector van %s naar %s (raster CRS)",
            gdf.crs.to_string(),
            raster_crs.to_string(),
        )
        gdf = gdf.to_crs(raster_crs)

    return gdf


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def compute_zonal_stats(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    stats: list[str],
    prefix: str,
) -> gpd.GeoDataFrame:
    """Calculate zonal statistics for every polygon in *gdf* against *raster_path*.

    Parameters
    ----------
    gdf:          GeoDataFrame with polygon geometries (must share CRS with raster).
    raster_path:  Path to a single-band GeoTIFF.
    stats:        List of statistics to compute, e.g. ["mean", "max", "std", "count"].
    prefix:       Column-name prefix, e.g. "hitte" → columns "hitte_mean", "hitte_max".

    Returns
    -------
    GeoDataFrame with new columns appended for each requested statistic.
    """
    if not raster_path.exists():
        raise FileNotFoundError(f"Rasterbestand niet gevonden: {raster_path}")

    log.info(
        "Bereken zonal stats [%s] voor %s (prefix='%s')",
        ", ".join(stats),
        raster_path.name,
        prefix,
    )

    # rasterstats expects a list of geometries in WKT or GeoJSON; passing the
    # GeoDataFrame directly is the most convenient approach.
    results = zonal_stats(
        gdf,
        str(raster_path),
        stats=stats,
        geojson_out=False,
        nodata=None,       # honour the raster's own nodata value
        all_touched=False, # only pixels whose centroid falls inside the polygon
    )

    # Attach results as new columns with the given prefix
    for stat in stats:
        col_name = f"{prefix}_{stat}"
        gdf[col_name] = [row.get(stat) for row in results]
        log.info("  Kolom toegevoegd: %s", col_name)

    return gdf


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_output(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write the enriched GeoDataFrame to a GeoPackage.

    Parameters
    ----------
    gdf:          Enriched GeoDataFrame.
    output_path:  Destination path (must end in .gpkg).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Schrijf resultaat naar %s", output_path)
    gdf.to_file(output_path, driver="GPKG", engine="pyogrio")
    log.info("  Klaar — %d rijen, %d kolommen", len(gdf), len(gdf.columns))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        "--raster",
        type=Path,
        nargs="+",
        required=True,
        metavar="TIF",
        help="Een of meer GeoTIFF-bestanden van de Klimaateffectatlas.",
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
        choices=["mean", "max", "min", "std", "count", "sum", "median", "range", "majority", "minority", "variety", "percentile_25", "percentile_75"],
        help=(
            "Te berekenen statistieken (standaard: mean max std count). "
            "Keuze uit: mean max min std count sum median range majority minority "
            "variety percentile_25 percentile_75."
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


def derive_prefix(raster_path: Path) -> str:
    """Derive a short column prefix from the raster file stem.

    Strips common suffixes like '_2023', '_v2', etc. and truncates to 20 chars
    so column names stay manageable.

    Examples
    --------
    'hitte_gevoelstemperatuur_2023.tif' → 'hitte_gevoelstemperatuur'
    'droogte_neerslagtekort.tif'        → 'droogte_neerslagtekort'
    """
    import re
    stem = raster_path.stem
    # Remove trailing year or version tags
    stem = re.sub(r"[_-]?(v\d+|\d{4})$", "", stem, flags=re.IGNORECASE)
    return stem[:20]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 1. Load CBS polygons
        gdf = load_wijken(args.wijken, args.layer, args.gemeente)

        # 2. Process each raster
        for raster_path in args.raster:
            prefix = derive_prefix(raster_path)
            # Reproject vector to raster CRS (in-place for this raster)
            gdf_aligned = align_crs(gdf, raster_path)
            gdf = compute_zonal_stats(gdf_aligned, raster_path, args.stats, prefix)

        # 3. Save enriched result
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