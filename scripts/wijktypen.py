"""
wijktypen.py
------------
Fetch wijktypen_buurten from the Klimaateffectatlas WFS and join to CBS GDF.

The WFS feature properties (confirmed via GetFeature inspection):
  BU_CODE     — CBS buurtcode, used for the attribute join
  Wijktype1   — dominant wijktype (e.g. "Tuinstad hoogbouw")
  Beoordelin  — confidence score for Wijktype1
  Wijktype2   — secondary wijktype
  Beoordeli2  — confidence score for Wijktype2
  WijktypeDe  — definitive/final wijktype label

Join strategy: attribute join on BU_CODE — fast and exact, no geometry needed.
Falls back to a spatial centroid join when BU_CODE is absent in *gdf*.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import requests

log = logging.getLogger(__name__)

WFS_URL   = (
    "https://cas.cloud.sogelink.com/public/data/org/gws/"
    "YWFMLMWERURF/kea_public/ows"
)
WFS_LAYER = "kea_public:wijktypen_buurten"
RD_CRS    = "EPSG:28992"

# Columns we want to pull from the WFS (subset to keep output clean)
WFS_COLS = {
    "BU_CODE":    "bu_code_wfs",   # only used for the join, dropped afterwards
    "WijktypeDe": "wijktype",      # definitive/dominant label
    "Wijktype1":  "wijktype_1",    # primary candidate
    "Beoordelin": "wijktype_1_score",
    "Wijktype2":  "wijktype_2",    # secondary candidate
    "Beoordeli2": "wijktype_2_score",
}


def _fetch_wfs_bbox(gdf: gpd.GeoDataFrame, timeout: int) -> gpd.GeoDataFrame:
    """Download wijktypen features clipped to *gdf* bounding box."""
    gdf_rd = gdf.to_crs(RD_CRS) if str(gdf.crs) != RD_CRS else gdf
    minx, miny, maxx, maxy = gdf_rd.total_bounds

    params = {
        "SERVICE":      "WFS",
        "VERSION":      "2.0.0",
        "REQUEST":      "GetFeature",
        "TYPENAMES":    WFS_LAYER,
        "SRSNAME":      RD_CRS,
        "BBOX":         f"{minx},{miny},{maxx},{maxy},{RD_CRS}",
        "OUTPUTFORMAT": "application/json",
    }

    log.info(
        "WFS ophalen: %s  bbox=(%.0f %.0f %.0f %.0f)",
        WFS_LAYER, minx, miny, maxx, maxy,
    )
    resp = requests.get(WFS_URL, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"WFS HTTP {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise RuntimeError(
            "WFS retourneerde 0 features voor dit gebied. "
            "Controleer of de bbox correct is."
        )

    wt = gpd.GeoDataFrame.from_features(features, crs=RD_CRS)
    log.info("  %d wijktype-features ontvangen", len(wt))
    return wt


def _attribute_join(
    gdf: gpd.GeoDataFrame,
    wt: gpd.GeoDataFrame,
    bu_col: str,
) -> gpd.GeoDataFrame:
    """Join on buurtcode — fast O(n) merge, no geometry operations needed."""
    # Rename WFS columns to clean output names; keep only what we need
    rename = {k: v for k, v in WFS_COLS.items() if k in wt.columns and k != "BU_CODE"}
    wt_slim = wt[["BU_CODE"] + list(rename.keys())].rename(columns=rename)

    result = gdf.merge(wt_slim, left_on=bu_col, right_on="BU_CODE", how="left")
    # Drop the redundant WFS buurtcode column
    if "BU_CODE" in result.columns and "BU_CODE" != bu_col:
        result = result.drop(columns=["BU_CODE"])

    matched = result["wijktype"].notna().sum()
    log.info("  Attribuut-join op '%s': %d/%d buurten gematcht", bu_col, matched, len(gdf))
    return result


def _spatial_join(
    gdf: gpd.GeoDataFrame,
    wt: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Fallback: join via centroid-in-polygon when no BU_CODE column exists."""
    log.info("  Geen BU_CODE kolom — spatial centroid join als fallback")

    gdf_rd   = gdf.to_crs(RD_CRS) if str(gdf.crs) != RD_CRS else gdf
    centroids = gdf_rd.copy()
    centroids["geometry"] = centroids.geometry.centroid

    rename = {k: v for k, v in WFS_COLS.items() if k in wt.columns and k != "BU_CODE"}
    wt_slim = wt[list(rename.keys()) + ["geometry"]].rename(columns=rename)

    joined = centroids[["geometry"]].sjoin(
        wt_slim, how="left", predicate="within"
    )

    for col in rename.values():
        if col in joined.columns:
            gdf = gdf.copy()
            gdf[col] = joined[col].values

    matched = gdf["wijktype"].notna().sum() if "wijktype" in gdf.columns else 0
    log.info("  Spatial join: %d/%d buurten gematcht", matched, len(gdf))
    return gdf


def join_wijktypen(
    gdf: gpd.GeoDataFrame,
    timeout: int = 30,
) -> gpd.GeoDataFrame:
    """Add wijktype columns to *gdf* via the Klimaateffectatlas WFS.

    Adds the following columns when available:
      wijktype          — dominant wijktype label (WijktypeDe)
      wijktype_1        — primary candidate (Wijktype1)
      wijktype_1_score  — confidence score (0–1)
      wijktype_2        — secondary candidate
      wijktype_2_score  — confidence score

    Parameters
    ----------
    gdf:     CBS neighbourhood GeoDataFrame filtered to one gemeente.
    timeout: HTTP request timeout in seconds.

    Returns
    -------
    GeoDataFrame with new wijktype columns appended.

    Raises
    ------
    RuntimeError  When the WFS request fails.
    """
    # 1. Download features for this municipality's bounding box
    wt = _fetch_wfs_bbox(gdf, timeout)

    # 2. Find BU_CODE column in gdf for the attribute join
    bu_col = next(
        (c for c in ("BU_CODE", "bu_code", "buurtcode", "BuurtCode") if c in gdf.columns),
        None,
    )

    if bu_col:
        result = _attribute_join(gdf, wt, bu_col)
    else:
        result = _spatial_join(gdf, wt)

    return result