"""
wijktypen.py
------------
Fetch the wijktypen_buurten layer from the Klimaateffectatlas WFS and join it
to the CBS GeoDataFrame as a new column.

The wijktypen layer assigns each neighbourhood to one of ~10 urban typology
classes (e.g. "Stadscentrum", "Naoorlogse wijk", "Groenstedelijk") based on
morphological and socioeconomic characteristics.

Source: Klimaateffectatlas / KEA public WFS (Sogelink)
Layer:  kea_public:wijktypen_buurten
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import requests

log = logging.getLogger(__name__)

WFS_URL        = (
    "https://cas.cloud.sogelink.com/public/data/org/gws/"
    "YWFMLMWERURF/kea_public/ows"
)
WFS_LAYER      = "kea_public:wijktypen_buurten"
WIJKTYPE_COL   = "wijktype"          # output column name in the enriched GDF
OUTPUT_CRS     = "EPSG:28992"


def fetch_wijktypen(
    gdf: gpd.GeoDataFrame,
    wfs_url: str = WFS_URL,
    layer: str = WFS_LAYER,
    timeout: int = 30,
) -> gpd.GeoDataFrame:
    """Download wijktypen polygons from WFS clipped to *gdf*'s bounding box.

    Parameters
    ----------
    gdf:     CBS neighbourhood GeoDataFrame (filtered to one gemeente).
    wfs_url: WFS base URL.
    layer:   WFS layer name.
    timeout: HTTP timeout in seconds.

    Returns
    -------
    GeoDataFrame with wijktypen polygons for the area, in EPSG:28992.

    Raises
    ------
    RuntimeError  When the WFS request fails or returns no features.
    """
    # Reproject to RD New so bbox coordinates are in metres
    gdf_rd = gdf.to_crs(OUTPUT_CRS) if str(gdf.crs) != OUTPUT_CRS else gdf
    minx, miny, maxx, maxy = gdf_rd.total_bounds

    params = {
        "SERVICE":      "WFS",
        "VERSION":      "2.0.0",
        "REQUEST":      "GetFeature",
        "TYPENAMES":    layer,
        "SRSNAME":      OUTPUT_CRS,
        "BBOX":         f"{minx},{miny},{maxx},{maxy},{OUTPUT_CRS}",
        "OUTPUTFORMAT": "application/json",
    }

    log.info("WFS ophalen: %s  bbox=(%.0f %.0f %.0f %.0f)", layer, minx, miny, maxx, maxy)
    resp = requests.get(wfs_url, params=params, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(
            f"WFS verzoek mislukt (HTTP {resp.status_code}): {resp.text[:200]}"
        )

    try:
        wt_gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs=OUTPUT_CRS)
    except Exception as exc:
        raise RuntimeError(f"Kan WFS-antwoord niet inlezen: {exc}") from exc

    if wt_gdf.empty:
        raise RuntimeError(
            f"WFS retourneerde geen features voor laag '{layer}' in dit gebied."
        )

    log.info("  %d wijktype-polygonen ontvangen", len(wt_gdf))
    return wt_gdf


def detect_type_column(wt_gdf: gpd.GeoDataFrame) -> str:
    """Return the column in *wt_gdf* that contains the wijktype classification.

    Tries common column name patterns; raises ValueError when none is found.
    """
    candidates = [
        "wijktype", "wijktype_naam", "type", "type_naam",
        "klasse", "typering", "omschrijving",
    ]
    cols_lower = {c.lower(): c for c in wt_gdf.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]

    # Fall back: first non-geometry string column
    for col in wt_gdf.columns:
        if col != "geometry" and wt_gdf[col].dtype == object:
            log.warning("Wijktype-kolom niet herkend — gebruik '%s'", col)
            return col

    raise ValueError(
        f"Geen wijktype-kolom gevonden in WFS-laag. "
        f"Beschikbare kolommen: {list(wt_gdf.columns)}"
    )


def join_wijktypen(
    gdf: gpd.GeoDataFrame,
    wfs_url: str = WFS_URL,
    layer: str = WFS_LAYER,
    output_col: str = WIJKTYPE_COL,
    timeout: int = 30,
) -> gpd.GeoDataFrame:
    """Add a *wijktype* column to *gdf* via a spatial join with the WFS layer.

    Strategy
    --------
    1. Download wijktypen polygons for the gemeente bounding box via WFS.
    2. Perform a spatial join (largest-overlap wins) to assign the dominant
       wijktype to each neighbourhood polygon in *gdf*.
    3. Return *gdf* with the new column appended.

    The largest-overlap strategy is used because CBS buurt boundaries do not
    always align exactly with wijktype polygon boundaries.

    Parameters
    ----------
    gdf:        CBS neighbourhood GeoDataFrame.
    output_col: Name of the new column in the output.

    Returns
    -------
    GeoDataFrame with *output_col* appended.
    """
    original_crs = gdf.crs

    # 1. Fetch
    wt_gdf = fetch_wijktypen(gdf, wfs_url=wfs_url, layer=layer, timeout=timeout)
    type_col = detect_type_column(wt_gdf)
    log.info("  Wijktype-kolom: '%s'  (%d unieke types)", type_col,
             wt_gdf[type_col].nunique())

    # 2. Align CRS
    gdf_rd = gdf.to_crs(OUTPUT_CRS) if str(gdf.crs) != OUTPUT_CRS else gdf

    # 3. Spatial join — for each buurt polygon, find overlapping wijktype polygons
    #    and keep the one with the largest intersection area (dominant type).
    joined = gpd.overlay(
        gdf_rd[["geometry"]].reset_index(),
        wt_gdf[["geometry", type_col]],
        how="intersection",
        keep_geom_type=False,
    )
    joined["_area"] = joined.geometry.area

    # Pick the wijktype with the largest overlap per buurt
    dominant = (
        joined.sort_values("_area", ascending=False)
        .drop_duplicates(subset=["index"])
        .set_index("index")[[type_col]]
        .rename(columns={type_col: output_col})
    )

    gdf = gdf.copy()
    gdf[output_col] = gdf.index.map(dominant[output_col])

    filled    = gdf[output_col].notna().sum()
    not_filled = gdf[output_col].isna().sum()
    log.info(
        "  Wijktype toegewezen: %d buurten ✓  %d zonder match",
        filled, not_filled,
    )

    # Restore original CRS if we reprojected
    if str(gdf.crs) != str(original_crs):
        gdf = gdf.to_crs(original_crs)

    return gdf