#!/usr/bin/env python3
"""
test_run.py
-----------
Quick smoke test / demo for the enrich_wijken pipeline.

Reads all settings from config.yaml by default.  Individual values can be
overridden via the LOCAL_OVERRIDES dict below or on the command line.

Raster sources — choose one or combine both:
  - LOCAL_RASTERS (from config.yaml)           → uses local GeoTIFF files
  - WMS (--wms / USE_WMS = True below)         → downloads from Klimaateffectatlas WMS

Run from the project root:
    python scripts/test_run.py

Use WMS instead of (or in addition to) local rasters:
    python scripts/test_run.py --wms

Override gemeente only:
    python scripts/test_run.py --gemeente Utrecht

Use a custom config:
    python scripts/test_run.py --config config.local.yaml

Verbose logging:
    python scripts/test_run.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"

# ===========================================================================
# Optional local overrides — set a value to override config.yaml.
# Leave as None to use config.yaml.
# ===========================================================================

LOCAL_OVERRIDES: dict[str, Any] = {
    "gemeente":  None,    # e.g. "Utrecht"
    "threshold": None,    # e.g. 35.0
    "normalize": False,

    # WMS settings
    # Set USE_WMS = True to download from WMS instead of (or alongside) local rasters.
    # The WMS layer and URL are taken from config.yaml (wms_layer / wms_url keys) or
    # the defaults in wms_utils.py when not present in config.
    "use_wms":       False,
    "wms_layer":     None,   # None → use DEFAULT_WMS_LAYER from wms_utils.py
    "wms_resolution": 10.0,  # metres per pixel
}

# ===========================================================================
# End of configuration
# ===========================================================================


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config; return {} when the file is absent."""
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_settings(
    cfg: dict[str, Any],
    overrides: dict[str, Any],
    use_wms: bool,
) -> dict[str, Any]:
    """Merge config.yaml with LOCAL_OVERRIDES and CLI flags.

    Parameters
    ----------
    cfg:      Parsed config.yaml dict.
    overrides: LOCAL_OVERRIDES dict (None values are skipped).
    use_wms:  When True, activate WMS even if not set in config or overrides.
    """
    root = Path(__file__).parent.parent

    def _abs(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    cbs_path  = _abs(cfg.get("cbs_path", "data/raw/wijkenbuurten_2023.gpkg"))
    cbs_layer = cfg.get("cbs_layer")
    gemeente  = overrides.get("gemeente") or cfg.get("gemeente")
    stats     = cfg.get("stats") or ["mean", "max", "std", "count"]
    threshold = overrides.get("threshold") if overrides.get("threshold") is not None else cfg.get("threshold")
    normalize = overrides.get("normalize") or False

    # Build local raster map
    raster_map: dict[str, Path] = {}
    for key, path_str in (cfg.get("rasters") or {}).items():
        raster_map[key] = _abs(path_str)

    # WMS settings
    from wms_utils import DEFAULT_WMS_LAYER, DEFAULT_WMS_URL
    wms_layer: str | None = None
    if use_wms or overrides.get("use_wms"):
        wms_layer = overrides.get("wms_layer") or cfg.get("wms_layer") or DEFAULT_WMS_LAYER
    wms_url        = cfg.get("wms_url", DEFAULT_WMS_URL)
    wms_resolution = overrides.get("wms_resolution") or 10.0

    out_dir = root / (cfg.get("output_dir") or "output")
    slug = (gemeente or "all").lower().replace(" ", "_")
    suffix = "_wms" if wms_layer and not raster_map else ""
    output = out_dir / f"test_{slug}_klimaat{suffix}.gpkg"

    return {
        "cbs_path":       cbs_path,
        "cbs_layer":      cbs_layer,
        "rasters":        raster_map,
        "wms_layer":      wms_layer,
        "wms_url":        wms_url,
        "wms_resolution": wms_resolution,
        "gemeente":       gemeente,
        "stats":          stats,
        "threshold":      threshold,
        "normalize":      normalize,
        "output":         output,
    }


def check_local_files(settings: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (missing_required, missing_optional) file paths.

    CBS file is required; individual rasters are optional (skipped with warning).
    WMS requires network access, not a local file.
    """
    missing_required = []
    missing_optional = []

    if not settings["cbs_path"].exists():
        missing_required.append(str(settings["cbs_path"]))

    for path in settings["rasters"].values():
        if not path.exists():
            missing_optional.append(str(path))

    return missing_required, missing_optional


def print_banner(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def print_settings(settings: dict[str, Any], config_path: Path) -> None:
    """Print a readable overview of resolved settings before running."""
    print_banner("klimaat-wijkprofielen-poc — test run")
    print(f"  Config        : {config_path}")
    print(f"  CBS bestand   : {settings['cbs_path'].name}")
    print(f"  CBS laag      : {settings['cbs_layer'] or 'eerste laag'}")
    print(f"  Gemeente      : {settings['gemeente'] or 'alle'}")
    print(f"  Stats         : {', '.join(settings['stats'])}")
    print(f"  Drempel       : {settings['threshold']}")
    print(f"  Normaliseer   : {settings['normalize']}")
    print()

    if settings["rasters"]:
        print("  Lokale rasters:")
        for key, path in settings["rasters"].items():
            status = "✓" if path.exists() else "⚠️  niet gevonden — wordt overgeslagen"
            print(f"    [{key}] {path.name}  {status}")

    if settings["wms_layer"]:
        print(f"\n  WMS-laag      : {settings['wms_layer']}")
        print(f"  WMS-URL       : {settings['wms_url']}")
        print(f"  WMS-resolutie : {settings['wms_resolution']} m/px")
        print("  (tijdelijk GeoTIFF wordt gedownload en na verwerking verwijderd)")

    if not settings["rasters"] and not settings["wms_layer"]:
        print("  ⚠️  Geen rasters geconfigureerd.")

    print(f"\n  Output        : {settings['output']}")


def print_summary(
    gdf,
    original_cols: set,
    elapsed: float,
    settings: dict[str, Any],
) -> None:
    added       = [c for c in gdf.columns if c not in original_cols and c != "geometry"]
    stat_cols   = [c for c in added if not c.endswith("_norm") and "pct_above" not in c]
    thresh_cols = [c for c in added if "pct_above" in c]
    norm_cols   = [c for c in added if c.endswith("_norm")]

    print_banner("Resultaten samenvatting")
    print(f"  Gemeente          : {settings['gemeente'] or 'alle'}")
    print(f"  Aantal rijen      : {len(gdf)}")
    print(f"  Verwerkingstijd   : {elapsed:.1f} seconden")
    print(f"  Uitvoerbestand    : {settings['output']}")

    if stat_cols:
        print(f"\n  Zonal stat-kolommen ({len(stat_cols)}):")
        for col in stat_cols:
            n     = gdf[col].notna().sum()
            vmin  = gdf[col].min()
            vmax  = gdf[col].max()
            vmean = gdf[col].mean()
            print(f"    {col:<40}  n={n:>3}  "
                  f"min={vmin:>8.2f}  max={vmax:>8.2f}  gem={vmean:>8.2f}")

    if thresh_cols:
        print(f"\n  Drempelwaarde-kolommen (>{settings['threshold']}) ({len(thresh_cols)}):")
        for col in thresh_cols:
            n     = gdf[col].notna().sum()
            vmean = gdf[col].mean()
            print(f"    {col:<40}  n={n:>3}  gemiddeld {vmean:.1f}% boven drempel")

    if norm_cols:
        print(f"\n  Genormaliseerde kolommen ({len(norm_cols)}):")
        for col in norm_cols:
            print(f"    {col}")

    print()


def run(config_path: Path, gemeente_override: str | None, use_wms: bool, verbose: bool) -> int:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {**LOCAL_OVERRIDES}
    if gemeente_override:
        overrides["gemeente"] = gemeente_override

    cfg = load_config(config_path)
    if not cfg:
        log.warning("config.yaml niet gevonden op %s — standaardwaarden gebruikt.", config_path)

    settings = resolve_settings(cfg, overrides, use_wms)
    print_settings(settings, config_path)

    # ── Check local files ────────────────────────────────────────────────────
    missing_req, missing_opt = check_local_files(settings)

    if missing_opt:
        log.warning("Volgende rasters niet gevonden — worden overgeslagen:")
        for p in missing_opt:
            log.warning("  • %s", p)
        settings["rasters"] = {k: v for k, v in settings["rasters"].items() if v.exists()}

    if missing_req:
        print_banner("❌  Ontbrekende bestanden")
        for p in missing_req:
            print(f"  • {p}")
        print(textwrap.dedent("""
          Pas config.yaml aan met de juiste bestandspaden, of raadpleeg
          README.md > "How to get the data" voor downloadinstructies.
        """))
        return 1

    if not settings["rasters"] and not settings["wms_layer"]:
        print_banner("❌  Geen rasters beschikbaar")
        print("  Voeg lokale rasters toe aan config.yaml, of gebruik --wms voor WMS-download.")
        return 1

    # ── Import pipeline ──────────────────────────────────────────────────────
    try:
        from enrich_wijken import load_wijken, compute_zonal_stats, save_output, _enrich_from_raster
        from utils import reproject_if_needed, calculate_percentage_above_threshold, normalize_column
        from wms_utils import TempRaster, download_wms_as_geotiff
    except ImportError as exc:
        log.error("Kan modules niet importeren: %s", exc)
        log.error("Installeer dependencies: pip install -r requirements.txt")
        return 2

    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        t_start = time.perf_counter()
        gdf = load_wijken(settings["cbs_path"], settings["cbs_layer"], settings["gemeente"])
        original_cols = set(gdf.columns)

        # 1. Local rasters
        for prefix, raster_path in settings["rasters"].items():
            gdf = _enrich_from_raster(
                gdf, raster_path, prefix,
                settings["stats"], settings["threshold"], settings["normalize"],
            )

        # 2. WMS raster (temp file, auto-deleted after the with-block)
        if settings["wms_layer"]:
            wms_prefix = settings["wms_layer"][:20]
            print_banner(f"WMS-download: {settings['wms_layer']}")
            print(f"  URL       : {settings['wms_url']}")
            print(f"  Resolutie : {settings['wms_resolution']} m/px")

            with TempRaster(suffix=f"_{settings['wms_layer']}.tif") as tmp_path:
                try:
                    download_wms_as_geotiff(
                        gdf,
                        wms_url=settings["wms_url"],
                        layer_name=settings["wms_layer"],
                        resolution_m=settings["wms_resolution"],
                        output_path=tmp_path,
                    )
                    gdf = _enrich_from_raster(
                        gdf, tmp_path, wms_prefix,
                        settings["stats"], settings["threshold"], settings["normalize"],
                    )
                    print(f"  ✓ WMS-verwerking geslaagd (tijdelijk bestand verwijderd)")
                except (ConnectionError, ValueError, RuntimeError) as exc:
                    log.error("WMS-verwerking mislukt: %s", exc)
                    log.error(
                        "Tips:\n"
                        "  • Controleer uw internetverbinding\n"
                        "  • Controleer of de WMS-URL bereikbaar is\n"
                        "  • Controleer de laagnaam met: python -c \"from wms_utils import *; "
                        "wms = connect_wms('%s'); print(list(wms.contents.keys()))\"",
                        settings["wms_url"],
                    )
                    if not settings["rasters"]:
                        return 1  # no fallback available
                    log.warning("Verdergaan zonder WMS-kolommen.")

        save_output(gdf, settings["output"])
        elapsed = time.perf_counter() - t_start
        print_summary(gdf, original_cols, elapsed, settings)
        print("✅  Test run geslaagd!")
        return 0

    except Exception as exc:
        log.exception("Pipeline mislukt: %s", exc)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test / demo voor de enrich_wijken pipeline."
    )
    parser.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG, metavar="YAML",
        help=f"Config-bestand (standaard: {DEFAULT_CONFIG.name}).")
    parser.add_argument("--gemeente", type=str, default=None, metavar="NAAM",
        help="Overschrijf de gemeente uit config.yaml.")
    parser.add_argument("--wms", action="store_true",
        help=(
            "Download de WMS-laag (hitteeiland_r_hitte) en gebruik die voor zonal stats. "
            "Kan gecombineerd worden met lokale rasters uit config.yaml."
        ))
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Toon DEBUG-logberichten.")
    args = parser.parse_args()
    return run(args.config, args.gemeente, args.wms, args.verbose)


if __name__ == "__main__":
    sys.exit(main())