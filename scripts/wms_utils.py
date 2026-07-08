"""
wms_utils.py
------------
Helpers for downloading a WMS layer as a georeferenced GeoTIFF so it can be
used as a raster input for zonal statistics.

The main entry point is :func:`download_wms_as_geotiff`.  It:

1. Connects to the WMS endpoint and validates that the requested layer exists.
2. Computes the bounding box of *gdf* (the filtered gemeente polygons) in the
   best available CRS, then expands it by a configurable buffer.
3. Requests a GetMap tile at a configurable resolution (default 50 m/px).
4. Writes the response as a fully georeferenced GeoTIFF and returns the path.

Bounding-box strategy
---------------------
We derive the bbox directly from the input GeoDataFrame rather than from the
WMS layer's advertised extent.  This ensures we download *only* the area that
matters for the analysis — typically a single municipality — which:

- keeps file sizes and download times small (a 50 m/px tile for Eindhoven
  is ~300 × 250 px rather than the full Netherlands);
- avoids memory issues with country-wide rasters;
- keeps rasterstats efficient because pixels outside the bbox are never loaded.

A buffer (default 500 m) is added around the tight bbox so that polygons
touching the edge get full pixel coverage and no edge artifacts appear in the
zonal statistics.

The caller is responsible for deleting the temp file when done; use the
:class:`TempRaster` context manager for automatic cleanup.
"""

from __future__ import annotations

import io
import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from owslib.wms import WebMapService

import geopandas as gpd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults (importable by other scripts for consistency)
# ---------------------------------------------------------------------------

#: WMS endpoint for the Klimaateffectatlas (Sogelink KEA public service).
DEFAULT_WMS_URL = (
    "https://cas.cloud.sogelink.com/public/data/org/gws/"
    "YWFMLMWERURF/kea_public/wms"
)

#: Default WMS layer — urban heat island (hitteeiland) raster.
DEFAULT_WMS_LAYER = "hitteeiland_r_hitte"

#: Default pixel size in metres.  50 m gives a good balance between detail
#: and download size for a single municipality (~200–400 px per side for most
#: Dutch gemeenten).  Use 10 m for higher-detail analysis.
DEFAULT_RESOLUTION_M = 50.0

#: Buffer added around the gemeente bounding box (metres).
#: Prevents edge-polygon artefacts and ensures full pixel coverage for all
#: neighbourhoods that touch the municipal boundary.
DEFAULT_BUFFER_M = 500.0

#: Hard cap on either raster dimension to prevent accidental huge downloads.
MAX_PIXELS = 4096

GEOTIFF_MIME  = "image/geotiff"
FALLBACK_MIMES = ["image/tiff", "image/png", "image/jpeg"]


# ---------------------------------------------------------------------------
# Context manager for temporary GeoTIFF
# ---------------------------------------------------------------------------

class TempRaster:
    """Context manager that yields the path to a temp GeoTIFF and cleans up on exit.

    Usage::

        with TempRaster(suffix="_wms_hitte.tif") as tmp_path:
            download_wms_as_geotiff(..., output_path=tmp_path)
            gdf = compute_zonal_stats(gdf, tmp_path, ...)
        # file is deleted here automatically
    """

    def __init__(self, suffix: str = "_wms.tif") -> None:
        self._suffix = suffix
        self._path: Path | None = None

    def __enter__(self) -> Path:
        import os
        fd, path_str = tempfile.mkstemp(suffix=self._suffix)
        os.close(fd)
        self._path = Path(path_str)
        return self._path

    def __exit__(self, *_) -> None:
        if self._path and self._path.exists():
            self._path.unlink()
            log.debug("Tijdelijk rasterbestand verwijderd: %s", self._path)


# ---------------------------------------------------------------------------
# WMS connection helpers
# ---------------------------------------------------------------------------

def connect_wms(url: str, timeout: int = 30) -> WebMapService:
    """Connect to a WMS endpoint and return the service object.

    Parameters
    ----------
    url:     WMS base URL (GetCapabilities is fetched automatically).
    timeout: HTTP request timeout in seconds.

    Raises
    ------
    ConnectionError  When the endpoint cannot be reached or returns an error.
    """
    log.info("Verbinding maken met WMS: %s", url)
    try:
        wms = WebMapService(url, version="1.1.1", timeout=timeout)
        log.info("  Verbonden — %d lagen beschikbaar", len(list(wms.contents)))
        return wms
    except Exception as exc:
        raise ConnectionError(f"Kan WMS niet bereiken op {url!r}: {exc}") from exc


def validate_layer(wms: WebMapService, layer_name: str) -> None:
    """Raise :class:`ValueError` when *layer_name* is not available on *wms*.

    Parameters
    ----------
    wms:        Connected WebMapService.
    layer_name: WMS layer identifier to verify.

    Raises
    ------
    ValueError  With a list of available layers to help the user fix the name.
    """
    if layer_name not in wms.contents:
        available = list(wms.contents.keys())
        # Show up to 15 layers; the full list can be retrieved separately.
        preview = ", ".join(available[:15])
        more    = f" … (+{len(available) - 15} meer)" if len(available) > 15 else ""
        raise ValueError(
            f"WMS-laag '{layer_name}' niet gevonden.\n"
            f"Beschikbare lagen: {preview}{more}\n"
            "Pas --wms-layer aan of bekijk alle lagen met:\n"
            f"  python -c \"from scripts.wms_utils import connect_wms; "
            f"wms=connect_wms('{wms.url}'); print('\\n'.join(wms.contents))\""
        )
    log.debug("WMS-laag '%s' gevonden.", layer_name)


def choose_crs(wms: WebMapService, layer_name: str, preferred: str = "EPSG:28992") -> str:
    """Return the best CRS for *layer_name* on *wms*.

    Prefers RD New (EPSG:28992) for Dutch data because it is metre-based,
    which makes resolution_m calculations straightforward.  Falls back to
    EPSG:4326 (WGS84) when RD New is not advertised.

    Parameters
    ----------
    preferred:  CRS string to try first.
    """
    layer     = wms.contents[layer_name]
    supported = {str(c).upper() for c in getattr(layer, "crsOptions", [])}
    log.debug("Ondersteunde CRS voor '%s': %s", layer_name, supported)

    for candidate in (preferred, "EPSG:28992", "EPSG:4326"):
        if candidate.upper() in supported:
            return candidate

    # Fall back to whatever the server advertises first
    fallback = str(next(iter(supported), preferred))
    log.warning("Voorkeurs-CRS niet beschikbaar — gebruik %s", fallback)
    return fallback


def pick_format(wms: WebMapService, layer_name: str) -> str:
    """Return the best available image format for GetMap requests.

    Prefers GeoTIFF (lossless, preserves float values).  Falls back to PNG
    (lossless integers) or JPEG as a last resort.

    Raises
    ------
    ValueError  When none of the preferred formats are supported.
    """
    try:
        getmap_op      = wms.getOperationByName("GetMap")
        server_formats = {f.lower() for f in getmap_op.formatOptions}
    except Exception:
        server_formats = set()

    for mime in [GEOTIFF_MIME] + FALLBACK_MIMES:
        if mime in server_formats:
            log.debug("Geselecteerd WMS-formaat: %s", mime)
            return mime

    log.warning("Kan ondersteunde formaten niet bepalen — probeer %s", GEOTIFF_MIME)
    return GEOTIFF_MIME


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def compute_buffered_bbox(
    gdf: gpd.GeoDataFrame,
    target_crs: str,
    buffer_m: float = DEFAULT_BUFFER_M,
) -> tuple[float, float, float, float]:
    """Return a buffered bounding box around *gdf* in *target_crs*.

    Strategy
    --------
    1. Reproject *gdf* to *target_crs* so buffer distances are in metres
       (assumes a metre-based CRS such as EPSG:28992).
    2. Compute the tight bbox of all polygons combined.
    3. Expand each side by *buffer_m* to avoid edge artefacts in zonal stats.

    The buffer matters because a neighbourhood polygon that touches the edge
    of the bbox would otherwise have pixels clipped on one side, giving a
    smaller sample and potentially a biased mean/max statistic.

    Parameters
    ----------
    gdf:        Input GeoDataFrame (gemeente polygons).
    target_crs: CRS of the WMS layer; ideally metre-based.
    buffer_m:   Expansion distance in metres (applied to all four sides).

    Returns
    -------
    (minx, miny, maxx, maxy) in *target_crs* coordinates.
    """
    # Reproject to the WMS CRS so we can work in consistent units
    if str(gdf.crs) != target_crs:
        gdf_proj = gdf.to_crs(target_crs)
    else:
        gdf_proj = gdf

    # Tight bbox of all municipality polygons combined
    minx, miny, maxx, maxy = gdf_proj.total_bounds

    log.debug(
        "Tight bbox (%s):    %.1f %.1f  →  %.1f %.1f  "
        "(%.1f × %.1f m)",
        target_crs, minx, miny, maxx, maxy,
        maxx - minx, maxy - miny,
    )

    # Expand by buffer on all four sides
    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m

    log.info(
        "  Bounding box met %.0f m buffer: %.1f %.1f  →  %.1f %.1f  "
        "(%.1f × %.1f m)",
        buffer_m, minx, miny, maxx, maxy,
        maxx - minx, maxy - miny,
    )

    return minx, miny, maxx, maxy


def compute_pixel_size(
    minx: float, miny: float, maxx: float, maxy: float,
    resolution_m: float,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Calculate (width_px, height_px) for a GetMap request.

    Clips dimensions to *max_pixels* on the longer side while preserving the
    aspect ratio, so we never accidentally request a multi-gigapixel image.

    Parameters
    ----------
    resolution_m: Desired pixel size in metres.  Smaller → more detail.
    max_pixels:   Hard cap on either dimension.

    Returns
    -------
    (width_px, height_px) as integers >= 1.
    """
    width_px  = max(1, math.ceil((maxx - minx) / resolution_m))
    height_px = max(1, math.ceil((maxy - miny) / resolution_m))

    if width_px > max_pixels or height_px > max_pixels:
        scale     = max_pixels / max(width_px, height_px)
        width_px  = max(1, int(width_px  * scale))
        height_px = max(1, int(height_px * scale))
        actual_res = (maxx - minx) / width_px
        log.warning(
            "Afbeelding geclipt naar %d × %d px (effectieve resolutie: %.1f m/px)",
            width_px, height_px, actual_res,
        )

    log.info("  Afbeeldingsgrootte: %d × %d px  (@%.0f m/px)", width_px, height_px, resolution_m)
    return width_px, height_px


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_wms_as_geotiff(
    gdf: gpd.GeoDataFrame,
    wms_url: str = DEFAULT_WMS_URL,
    layer_name: str = DEFAULT_WMS_LAYER,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    buffer_m: float = DEFAULT_BUFFER_M,
    output_path: Path | None = None,
    wms_timeout: int = 60,
) -> Path:
    """Download a WMS layer clipped to *gdf*'s (buffered) bounding box as a GeoTIFF.

    The bounding box is derived from the *gdf* polygons (e.g. the filtered
    gemeente neighbourhoods), not from the WMS layer's full advertised extent.
    This ensures only the relevant area is downloaded.

    Parameters
    ----------
    gdf:
        GeoDataFrame whose bounding box defines the download area.  Pass the
        *filtered* CBS neighbourhood polygons (gemeente already applied) so
        the download window matches the analysis area exactly.
    wms_url:
        WMS base URL.  Defaults to the Klimaateffectatlas Sogelink endpoint.
    layer_name:
        WMS layer identifier.  Defaults to ``"hitteeiland_r_hitte"``.
    resolution_m:
        Target pixel size in metres.  Default 50 m ≈ 200–400 px for most
        Dutch gemeenten.  Use 10 m for detailed neighbourhood analysis.
    buffer_m:
        Buffer around the municipality bbox in metres (default 500 m).
        Prevents edge artefacts in zonal statistics for border polygons.
    output_path:
        Where to write the GeoTIFF.  When None a temp file is created; use
        :class:`TempRaster` as a context manager for automatic cleanup.
    wms_timeout:
        HTTP timeout in seconds.

    Returns
    -------
    :class:`Path` to the written GeoTIFF file.

    Raises
    ------
    ConnectionError  When the WMS endpoint cannot be reached.
    ValueError       When *layer_name* is not available on the server.
    RuntimeError     When the GetMap response cannot be decoded or written.
    """
    if gdf.empty:
        raise ValueError("Lege GeoDataFrame doorgegeven — kan geen bbox berekenen.")

    # ── 1. Connect & validate ────────────────────────────────────────────────
    wms = connect_wms(wms_url, timeout=wms_timeout)
    validate_layer(wms, layer_name)

    # ── 2. CRS selection ─────────────────────────────────────────────────────
    # Prefer metre-based RD New so resolution_m maps directly to pixel counts.
    crs_str = choose_crs(wms, layer_name)
    log.info("WMS CRS: %s", crs_str)

    # ── 3. Bbox from gemeente polygons + buffer ───────────────────────────────
    # Using the municipality's own polygons as the bbox source means we never
    # download more than we need, keeping tiles small and processing fast.
    minx, miny, maxx, maxy = compute_buffered_bbox(gdf, crs_str, buffer_m)

    # ── 4. Pixel dimensions ──────────────────────────────────────────────────
    width_px, height_px = compute_pixel_size(minx, miny, maxx, maxy, resolution_m)

    # ── 5. Image format ──────────────────────────────────────────────────────
    img_format = pick_format(wms, layer_name)

    # ── 6. GetMap request ────────────────────────────────────────────────────
    log.info(
        "GetMap: laag='%s'  bbox=(%.1f %.1f %.1f %.1f)  "
        "size=%d×%d  formaat=%s",
        layer_name, minx, miny, maxx, maxy, width_px, height_px, img_format,
    )
    try:
        response  = wms.getmap(
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

    log.info("  Ontvangen: %d bytes (%.1f KB)", len(raw_bytes), len(raw_bytes) / 1024)

    # ── 7. Write to GeoTIFF ──────────────────────────────────────────────────
    if output_path is None:
        import os
        fd, tmp = tempfile.mkstemp(suffix=f"_{layer_name}.tif")
        os.close(fd)
        output_path = Path(tmp)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if img_format in (GEOTIFF_MIME, "image/tiff"):
        # Response is already a (Geo)TIFF — write raw bytes then ensure
        # the CRS and geotransform are correct (some servers omit them).
        output_path.write_bytes(raw_bytes)
        _ensure_georeferenced(output_path, minx, miny, maxx, maxy, crs_str)
    else:
        # PNG / JPEG response — decode pixels and write new georeferenced GeoTIFF.
        _write_georeferenced_from_image(
            raw_bytes, output_path, minx, miny, maxx, maxy, crs_str,
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
) -> None:
    """Overwrite the CRS and transform of an existing GeoTIFF if they are missing.

    Some WMS servers return a valid GeoTIFF byte stream but omit the embedded
    georeferencing.  This function opens the file in read-write mode and injects
    the correct values computed from the GetMap bbox.
    """
    with rasterio.open(path, "r+") as ds:
        missing_crs       = ds.crs is None
        missing_transform = (
            ds.transform is None
            or ds.transform == rasterio.transform.IDENTITY
        )
        if missing_crs or missing_transform:
            log.debug(
                "Georeferentie ontbreekt in WMS-antwoord — "
                "CRS=%s, transform wordt ingesteld.", crs_str
            )
            if missing_crs:
                ds.crs = rasterio.CRS.from_string(crs_str)
            if missing_transform:
                ds.transform = from_bounds(minx, miny, maxx, maxy, ds.width, ds.height)
        # Ensure nodata=0 is set for uint8 rasters (KEA classified layers use 0 for water/outside)
        if ds.nodata is None and ds.dtypes[0] == "uint8":
            ds.nodata = 0
            log.debug("Nodata=0 ingesteld voor uint8 raster")


def _write_georeferenced_from_image(
    raw_bytes: bytes,
    output_path: Path,
    minx: float, miny: float, maxx: float, maxy: float,
    crs_str: str,
) -> None:
    """Decode a PNG/JPEG byte string and write a georeferenced single-band GeoTIFF.

    Only the first band is kept because rasterstats expects a single-band raster.
    RGB imagery (as returned by some WMS servers for visualisation layers) is
    reduced to band 1; for categorical rasters this is fine because all bands
    are identical in those cases.

    Requires Pillow (``pip install Pillow``).

    Raises
    ------
    RuntimeError  When Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "Pillow is vereist voor PNG/JPEG WMS-antwoorden. "
            "Installeer met: pip install Pillow"
        )

    img = Image.open(io.BytesIO(raw_bytes))
    arr = np.array(img)

    # Reduce to a single band for zonal statistics
    if arr.ndim == 3:
        log.debug(
            "WMS-antwoord heeft %d banden — alleen band 1 wordt gebruikt.", arr.shape[2]
        )
        arr = arr[:, :, 0]

    height, width = arr.shape
    transform     = from_bounds(minx, miny, maxx, maxy, width, height)

    # Attempt to detect the nodata value from the image:
    # WMS layers like hitteeiland use 0 for water/outside-area pixels.
    # We preserve the nodata tag so rasterstats can exclude these pixels.
    _nodata_val = 0 if arr.min() == 0 and arr.dtype == np.uint8 else None

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
        nodata=_nodata_val,
    ) as ds:
        ds.write(arr, 1)