"""
utils.py
--------
Shared helper functions for the klimaat-wijkprofielen pipeline.

All functions are stateless and side-effect free so they can be reused
in scripts, notebooks, and tests without modification.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def reproject_if_needed(
    gdf: gpd.GeoDataFrame,
    raster_path: str | Path,
) -> gpd.GeoDataFrame:
    """Return *gdf* reprojected to the CRS of *raster_path* if they differ.

    rasterstats requires vector and raster to share the same coordinate
    reference system.  This function handles the three possible states:
    - Vector has no CRS  → assign the raster CRS and log a warning.
    - CRS matches        → return the original object unchanged.
    - CRS differs        → reproject and return the new object.

    Parameters
    ----------
    gdf:
        Input GeoDataFrame with polygon or multipolygon geometries.
    raster_path:
        Path to a GeoTIFF whose CRS is used as the reprojection target.

    Returns
    -------
    GeoDataFrame in the raster CRS.  May be the same object when CRS
    already matches (no copy is made).

    Raises
    ------
    FileNotFoundError
        When *raster_path* does not exist.
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Rasterbestand niet gevonden: {raster_path}")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if gdf.crs is None:
        import warnings
        warnings.warn(
            f"Vector heeft geen CRS — raster-CRS wordt aangenomen ({raster_crs}).",
            stacklevel=2,
        )
        return gdf.set_crs(raster_crs)

    if gdf.crs == raster_crs:
        return gdf

    return gdf.to_crs(raster_crs)


# ---------------------------------------------------------------------------
# Threshold statistics
# ---------------------------------------------------------------------------

def calculate_percentage_above_threshold(
    gdf: gpd.GeoDataFrame,
    raster_path: str | Path,
    threshold: float,
    prefix: str,
) -> gpd.GeoDataFrame:
    """Add a column with the percentage of raster pixels above *threshold* per polygon.

    For each polygon the function counts valid (non-nodata) pixels and the
    subset of those that exceed *threshold*, then stores the ratio as a
    percentage in a new column ``{prefix}_pct_above_{threshold}``.

    Parameters
    ----------
    gdf:
        GeoDataFrame whose geometry is used for the zonal query.
        Must already share the CRS of *raster_path* — call
        :func:`reproject_if_needed` first when unsure.
    raster_path:
        Path to a single-band GeoTIFF.
    threshold:
        Numeric boundary.  Pixels with value **strictly greater than**
        this number are counted as "above threshold".
    prefix:
        Short identifier for the raster theme, e.g. ``"hitte"``.
        Used as the column-name prefix.

    Returns
    -------
    GeoDataFrame with one new column appended:
    ``{prefix}_pct_above_{threshold}``  — float in [0, 100] or NaN when
    the polygon contains no valid pixels.

    Notes
    -----
    - Nodata pixels are excluded from both the numerator and denominator.
    - The column value is rounded to two decimal places.
    """
    raster_path = Path(raster_path)

    # Use rasterstats to retrieve per-pixel arrays via mini_rasters=False;
    # the 'values' entry from gen_zonal_stats gives us raw pixel arrays.
    results = zonal_stats(
        gdf,
        str(raster_path),
        stats=[],           # no standard stats needed
        add_stats={},
        raster_out=True,    # include the clipped raster array per polygon
        nodata=None,
    )

    col_name = f"{prefix}_pct_above_{threshold}"
    percentages: list[float | None] = []

    for row in results:
        mini = row.get("mini_raster_array")
        if mini is None:
            # Fallback: raster_out stores the array under 'mini_raster_array'
            percentages.append(None)
            continue

        arr = np.ma.compressed(mini)   # flatten, remove masked/nodata pixels
        if arr.size == 0:
            percentages.append(None)
        else:
            pct = float(np.sum(arr > threshold) / arr.size * 100)
            percentages.append(round(pct, 2))

    gdf = gdf.copy()
    gdf[col_name] = percentages
    return gdf


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a min-max normalised copy of *column* scaled to [0, 1].

    Useful for creating composite climate risk scores from columns that
    have different units or value ranges.

    Parameters
    ----------
    df:
        DataFrame (or GeoDataFrame) containing *column*.
    column:
        Name of the numeric column to normalise.

    Returns
    -------
    pd.Series with the same index as *df*, values in [0.0, 1.0].
    Returns a Series of NaN when all values are identical (zero range).

    Raises
    ------
    KeyError
        When *column* is not present in *df*.
    TypeError
        When *column* contains non-numeric data.

    Examples
    --------
    >>> gdf["hitte_mean_norm"] = normalize_column(gdf, "hitte_mean")
    """
    if column not in df.columns:
        raise KeyError(f"Kolom '{column}' niet gevonden in DataFrame.")

    series = pd.to_numeric(df[column], errors="raise")
    col_min = series.min()
    col_max = series.max()

    if col_max == col_min:
        import warnings
        warnings.warn(
            f"Kolom '{column}' heeft een bereik van 0 — normalisatie retourneert NaN.",
            stacklevel=2,
        )
        return pd.Series(np.nan, index=df.index, name=column)

    return (series - col_min) / (col_max - col_min)