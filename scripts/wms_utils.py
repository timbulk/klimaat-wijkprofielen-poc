"""
wms_utils.py
------------
Helpers for downloading a WMS layer as a georeferenced GeoTIFF so it can be
used as a raster input for zonal statistics.

The main entry point is :func:`download_wms_as_geotiff`.  It:

1. Connects to the WMS and validates that the requested layer exists.
2. Reprojects the bounding box of *gdf* to the WMS-supported CRS.
3. Requests a GetMap tile at a configurable resolution.
4. Writes the response (GeoTIFF or PNG/JPEG fallback) as a georeferenced
   GeoTIFF to a temporary file and returns the path.

The caller is responsible for deleting the temp file when done (or use the
returned :class:`TempRaster` context manager).
"""

from __future__ import annotations

import io
import logging
import math
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from owslib.wms import WebMapService

import geopandas as gpd

log = logging.getLogger(__name__)

# Default WMS for Klimaateffectatlas (Sogelink / KEA public service)
DEFAULT_WMS_URL = (
    "https://cas.cloud.sogelink.com/public/data/org/gws/"
    "YWFMLMWERURF/kea_public/wms"
)

# Preferred layer for urban heat island (hitteeiland) analysis
DEFAULT_WMS_LAYER = "hitteeiland_r_hitte"

# Resolution in metres per pixel when no explicit value is given.
# 10 m gives a good balance between detail and download size for a municipality.
DEFAULT_RESOLUTION_M = 10.0

# Maximum raster dimensions to avoid accidentally requesting a giant image.
MAX_PIXELS = 4096

# Preferred output format — most WMS servers that carry raster data support this.
GEOTIFF_MIME = "image/geotiff"
FALLBACK_MIMES = ["image/tiff", "image/png", "image/jpeg"]


# ---------------------------------------------------------------------------
# Context manager for temporary GeoTIFF
# ---------------------------------------------------------------------------

class TempRaster:
    """Context manager that yields the path to a temp GeoTIFF and cleans up.

    Usage::

        with TempRaster(suffix="_wms.tif") as tmp_path:
            download_wms_as_geotiff(..., output_path=tmp_path)
            gdf = compute_zonal_stats(gdf, tmp_path, ...)
        # file is deleted here
    """

    def __init__(self, suffix: str = "_wms.tif") -> None:
        self._suffix = suffix
        self._path: Path | None = None

    def __enter__(self) -> Path:
        fd, path_str = tempfile.mkstemp(suffix=self._suffix)
        import os
        os.close(fd)
        self._path = Path(path_str)
        return self._path

    def __exit__(self, *_) -> None:
        if self._path and self._path.exists():
            self._path.unlink()
            log.debug("Tijdelijk rasterbestand verwijderd: %s", self._path)


# ---------------------------------------------------------------------------
# WMS helpers
# ---------------------------------------------------------------------------

def connect_wms(url: str, timeout: int = 30) -> WebMapService:
    """Connect to a WMS endpoint and return the service object.

    Parameters
    ----------
    url:     WMS base URL (GetCapabilities is fetched automatically).
    timeout: Request timeout in seconds.

    Returns
    -------
    :class:`owslib.wms.WebMapService` instance.

    Raises
    ------
    ConnectionError  When the WMS cannot be reached or returns an error.
    """
    log.info("Verbinding maken met WMS: %s", url)
    try:
        wms = WebMapService(url, version="1.1.1", timeout=timeout)
        log.info("  Verbonden — %d lagen beschikbaar", len(list(wms.contents)))
        return wms
    except Exception as exc:
        raise ConnectionError(
            f"Kan WMS niet bereiken op {url!r}: {exc}"
        ) from exc


def validate_layer(wms: WebMapService, layer_name: str) -> None:
    """Raise :class:`ValueError` when *layer_name* is not in *wms*.

    Parameters
    ----------
    wms:        Connected WebMapService.
    layer_name: WMS layer identifier to check.

    Raises
    ------
    ValueError  When the layer is absent, with a list of available layers.
    """
    available = list(wms.contents.keys())
    if layer_name not in wms.contents:
        suggestions = ", ".join(available[:10])
        raise ValueError(
            f"WMS-laag '{layer_name}' niet gevonden.\n"
            f"Beschikbare lagen (eerste 10): {suggestions}\n"
            f"Pas --wms-layer aan of bekijk de volledige lijst op de WMS-endpoint."
        )
    log.debug("WMS-laag '%s' bestaat.", layer_name)


def choose_crs(wms: WebMapService, layer_name: str, preferred: str = "EPSG:28992") -> str:
    """Return the best available CRS for *layer_name*.

    Prefers *preferred* (RD New) for Dutch data, falls back to EPSG:4326.

    Parameters
    ----------
    wms:        Connected WebMapService.
    layer_name: WMS layer identifier.
    preferred:  CRS to use when the server supports it.

    Returns
    -------
    CRS string such as ``"EPSG:28992"``.
    """
    layer = wms.contents[layer_name]
    supported = {str(c).upper() for c in getattr(layer, "crsOptions", [])}
    log.debug("Ondersteunde CRS voor '%s': %s", layer_name, supported)

    if preferred.upper() in supported:
        return preferred
    if "EPSG:28992" in supported:
        return "EPSG:28992"
    if "EPSG:4326" in supported:
        return "EPSG:4326"
    # Return whatever is first
    return str(next(iter(supported), preferred))


def pick_format(wms: WebMapService, layer_name: str) -> str:
    """Return the best available image format for GetMap requests.

    Prefers GeoTIFF for lossless raster values; falls back to PNG or JPEG.

    Parameters
    ----------
    wms:        Connected WebMapService.
    layer_name: WMS layer identifier.

    Returns
    -------
    MIME type string, e.g. ``"image/geotiff"``.

    Raises
    ------
    ValueError  When none of the preferred formats are supported.
    """
    # getOperationByName returns an OperationMetadata object
    try:
        getmap_op = wms.getOperationByName("GetMap")
        server_formats = {f.lower() for f in getmap_op.formatOptions}
    except Exception:
        server_formats = set()

    for mime in [GEOTIFF_MIME] + FALLBACK_MIMES:
        if mime in server_formats:
            log.debug("Geselecteerd formaat: %s", mime)
            return mime

    # If server didn't advertise formats, try GeoTIFF anyway
    log.warning(
        "Kan ondersteunde formaten niet bepalen — probeer %s", GEOTIFF_MIME
    )
    return GEOTIFF_MIME


def _bbox_for_gdf(gdf: gpd.GeoDataFrame, target_crs: str) -> tuple[float, float, float, float]:
    """Return the (minx, miny, maxx, maxy) bounding box of *gdf* in *target_crs*."""
    gdf_proj = gdf.to_crs(target_crs) if str(gdf.crs) != target_crs else gdf
    return tuple(gdf_proj.total_bounds)  # type: ignore[return-value]


def _clamp_size(width: int, height: int, max_px: int = MAX_PIXELS) -> tuple[int, int]:
    """Scale down (width, height) so neither dimension exceeds *max_px*."""
    if width <= max_px and height <= max_px:
        return width, height
    scale = max_px / max(width, height)
    return max(1, int(width * scale)), max(1, int(height * scale))


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_wms_as_geotiff(
    gdf: gpd.GeoDataFrame,
    wms_url: str = DEFAULT_WMS_URL,
    layer_name: str = DEFAULT_WMS_LAYER,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    output_path: Path | None = None,
    wms_timeout: int = 60,
) -> Path:
    """Download a WMS layer clipped to *gdf*'s bounding box as a GeoTIFF.

    The resulting file is georeferenced and can be passed directly to
    :func:`rasterstats.zonal_stats` or :func:`utils.reproject_if_needed`.

    Parameters
    ----------
    gdf:
        GeoDataFrame whose bounding box defines the area to download.
        Typically the filtered CBS neighbourhood polygons.
    wms_url:
        WMS base URL.  Defaults to the Klimaateffectatlas Sogelink endpoint.
    layer_name:
        WMS layer identifier.  Defaults to ``"hitteeiland_r_hitte"``.
    resolution_m:
        Target pixel size in metres (ignored when the WMS CRS is geographic).
        Smaller values → more detail but larger downloads.  Default: 10 m.
    output_path:
        Where to write the GeoTIFF.  When None a temp file is created; the
        caller is then responsible for deleting it.  Use :class:`TempRaster`
        as a context manager for automatic cleanup.
    wms_timeout:
        HTTP timeout in seconds for the WMS requests.

    Returns
    -------
    Path to the written GeoTIFF.

    Raises
    ------
    ConnectionError  When the WMS endpoint cannot be reached.
    ValueError       When *layer_name* is not available on the server.
    RuntimeError     When the GetMap response cannot be written as a GeoTIFF.
    """
    # 1. Connect and validate
    wms = connect_wms(wms_url, timeout=wms_timeout)
    validate_layer(wms, layer_name)

    # 2. Choose CRS — prefer RD New (EPSG:28992) for Dutch data
    crs_str = choose_crs(wms, layer_name)
    log.info("WMS-laag '%s'  CRS: %s", layer_name, crs_str)

    # 3. Compute bounding box in the chosen CRS
    minx, miny, maxx, maxy = _bbox_for_gdf(gdf, crs_str)
    log.debug("Bounding box (%s): %.2f %.2f %.2f %.2f", crs_str, minx, miny, maxx, maxy)

    # 4. Calculate pixel dimensions from desired resolution
    width_m  = maxx - minx
    height_m = maxy - miny
    width_px  = max(1, math.ceil(width_m  / resolution_m))
    height_px = max(1, math.ceil(height_m / resolution_m))
    width_px, height_px = _clamp_size(width_px, height_px)
    log.info("  Afbeeldingsgrootte: %d × %d px (@%.0f m/px)", width_px, height_px, resolution_m)

    # 5. Choose image format
    img_format = pick_format(wms, layer_name)

    # 6. GetMap request
    log.info("  GetMap aanvragen voor laag '%s' …", layer_name)
    try:
        response = wms.getmap(
            layers=[layer_name],
            srs=crs_str,
            bbox=(minx, miny, maxx, maxy),
            size=(width_px, height_px),
            format=img_format,
            transparent=False,
        )
        raw_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(
            f"WMS GetMap mislukt voor laag '{layer_name}': {exc}"
        ) from exc

    log.info("  Ontvangen: %d bytes", len(raw_bytes))

    # 7. Write to GeoTIFF
    if output_path is None:
        fd, tmp = tempfile.mkstemp(suffix=f"_{layer_name}.tif")
        import os; os.close(fd)
        output_path = Path(tmp)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if img_format in (GEOTIFF_MIME, "image/tiff"):
        # Response is already a (Geo)TIFF — write raw bytes, then verify/re-georeference
        output_path.write_bytes(raw_bytes)
        _ensure_georeferenced(output_path, minx, miny, maxx, maxy, crs_str, width_px, height_px)
    else:
        # PNG / JPEG — decode pixel array and write a new georeferenced GeoTIFF
        _write_georeferenced_from_image(
            raw_bytes, img_format, output_path,
            minx, miny, maxx, maxy, crs_str,
        )

    log.info("WMS-raster opgeslagen: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# GeoTIFF writing helpers
# ---------------------------------------------------------------------------

def _ensure_georeferenced(
    path: Path,
    minx: float, miny: float, maxx: float, maxy: float,
    crs_str: str,
    width: int, height: int,
) -> None:
    """Open an existing (Geo)TIFF and write CRS + transform when absent."""
    with rasterio.open(path, "r+") as ds:
        needs_crs       = ds.crs is None
        needs_transform = ds.transform == rasterio.transform.IDENTITY or ds.transform is None

        if needs_crs or needs_transform:
            log.debug("Geo-referentie ontbreekt in WMS-antwoord — wordt toegevoegd.")
            ds.crs = rasterio.CRS.from_string(crs_str)
            ds.transform = from_bounds(minx, miny, maxx, maxy, ds.width, ds.height)


def _write_georeferenced_from_image(
    raw_bytes: bytes,
    mime: str,
    output_path: Path,
    minx: float, miny: float, maxx: float, maxy: float,
    crs_str: str,
) -> None:
    """Decode a PNG/JPEG byte string and write a georeferenced single-band GeoTIFF.

    For multi-band images (RGB) only band 1 is written because zonal_stats
    expects a single-band raster.
    """
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "Pillow is vereist om PNG/JPEG WMS-antwoorden te verwerken. "
            "Installeer het met: pip install Pillow"
        )

    img = Image.open(io.BytesIO(raw_bytes))
    arr = np.array(img)

    # Use only the first band for raster statistics
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    height, width = arr.shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=arr.dtype,
        crs=rasterio.CRS.from_string(crs_str),
        transform=transform,
    ) as ds:
        ds.write(arr, 1)