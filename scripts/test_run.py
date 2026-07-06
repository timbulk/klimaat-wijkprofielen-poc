#!/usr/bin/env python3
"""
test_run.py
-----------
Quick smoke test / demo run for the enrich_wijken pipeline.

By default this script reads all settings from config.yaml in the project root.
You can also override individual values below or pass --config to point at a
different YAML file.

Run from the project root:
    python scripts/test_run.py

Use a custom config:
    python scripts/test_run.py --config config.local.yaml

Override gemeente only:
    python scripts/test_run.py --gemeente Utrecht

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

# Make sure sibling scripts are importable
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"

# ===========================================================================
# Optional local overrides
# Set a value here to override config.yaml without editing it.
# Leave as None to use config.yaml.
# ===========================================================================

LOCAL_OVERRIDES: dict[str, Any] = {
    "gemeente":  None,   # e.g. "Utrecht" — overrides config.yaml
    "threshold": None,   # e.g. 35.0
    "normalize": False,  # set True to add normalised columns
}

# ===========================================================================
# End of configuration
# ===========================================================================


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from *config_path*.

    Returns an empty dict when the file does not exist so the rest of the
    script can always treat the result as a dict.
    """
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_settings(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge config.yaml values with LOCAL_OVERRIDES.

    *overrides* values of None are ignored so unset keys fall through to cfg.

    Parameters
    ----------
    cfg:       Parsed config.yaml dict.
    overrides: LOCAL_OVERRIDES dict (None values are skipped).

    Returns
    -------
    Resolved settings dict ready for the pipeline.
    """
    root = Path(__file__).parent.parent

    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    cbs_path  = _abs(cfg.get("cbs_path", "data/raw/wijkenbuurten_2023.gpkg"))
    cbs_layer = cfg.get("cbs_layer")
    gemeente  = overrides.get("gemeente") or cfg.get("gemeente")
    stats     = cfg.get("stats") or ["mean", "max", "std", "count"]
    threshold = overrides.get("threshold") if overrides.get("threshold") is not None else cfg.get("threshold")
    normalize = overrides.get("normalize") or False

    # Build raster map from config
    raster_map: dict[str, Path] = {}
    for key, path_str in (cfg.get("rasters") or {}).items():
        raster_map[key] = _abs(path_str)

    # Derive output path
    out_dir = root / (cfg.get("output_dir") or "output")
    slug = (gemeente or "all").lower().replace(" ", "_")
    output = out_dir / f"test_{slug}_klimaat.gpkg"

    return {
        "cbs_path":  cbs_path,
        "cbs_layer": cbs_layer,
        "rasters":   raster_map,
        "gemeente":  gemeente,
        "stats":     stats,
        "threshold": threshold,
        "normalize": normalize,
        "output":    output,
    }


def check_files(settings: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (ok, missing_paths) for required input files."""
    required = [settings["cbs_path"]] + list(settings["rasters"].values())
    missing = [str(p) for p in required if not p.exists()]
    return len(missing) == 0, missing


def print_banner(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_settings(settings: dict[str, Any], config_path: Path) -> None:
    """Print a readable overview of resolved settings."""
    print_banner("klimaat-wijkprofielen-poc — test run")
    print(f"  Config     : {config_path}")
    print(f"  CBS bestand: {settings['cbs_path'].name}")
    print(f"  CBS laag   : {settings['cbs_layer'] or 'eerste laag'}")
    print(f"  Gemeente   : {settings['gemeente'] or 'alle'}")
    for key, path in settings["rasters"].items():
        exists = "✓" if path.exists() else "⚠️  niet gevonden"
        print(f"  Raster [{key}]: {path.name}  {exists}")
    print(f"  Stats      : {', '.join(settings['stats'])}")
    print(f"  Drempel    : {settings['threshold']}")
    print(f"  Normaliseer: {settings['normalize']}")
    print(f"  Output     : {settings['output']}")


def print_summary(gdf, original_cols: set, elapsed: float, settings: dict[str, Any]) -> None:
    """Print a tabular summary of all enrichment columns."""
    added = [c for c in gdf.columns if c not in original_cols and c != "geometry"]
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
            n    = gdf[col].notna().sum()
            vmin = gdf[col].min()
            vmax = gdf[col].max()
            vmean= gdf[col].mean()
            print(f"    {col:<38}  n={n:>3}  "
                  f"min={vmin:>8.2f}  max={vmax:>8.2f}  gem={vmean:>8.2f}")

    if thresh_cols:
        thr = settings["threshold"]
        print(f"\n  Drempelwaarde-kolommen (>{thr}) ({len(thresh_cols)}):")
        for col in thresh_cols:
            n    = gdf[col].notna().sum()
            vmean= gdf[col].mean()
            print(f"    {col:<38}  n={n:>3}  gemiddeld {vmean:.1f}% boven drempel")

    if norm_cols:
        print(f"\n  Genormaliseerde kolommen ({len(norm_cols)}):")
        for col in norm_cols:
            print(f"    {col}")

    print()


def run(config_path: Path, gemeente_override: str | None, verbose: bool) -> int:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Merge LOCAL_OVERRIDES with any CLI gemeente override
    overrides = {**LOCAL_OVERRIDES}
    if gemeente_override:
        overrides["gemeente"] = gemeente_override

    cfg = load_config(config_path)
    if not cfg:
        log.warning(
            "config.yaml niet gevonden op %s — gebruik standaardwaarden.", config_path
        )

    settings = resolve_settings(cfg, overrides)
    print_settings(settings, config_path)

    # ── Check required files ─────────────────────────────────────────────────
    ok, missing = check_files(settings)
    if not ok:
        # Skip missing rasters gracefully; only fail on missing CBS file
        missing_cbs   = [p for p in missing if "wijken" in p.lower() or p == str(settings["cbs_path"])]
        missing_rasters = [p for p in missing if p not in missing_cbs]

        if missing_rasters:
            log.warning("Volgende rasters niet gevonden en worden overgeslagen:")
            for p in missing_rasters:
                log.warning("  • %s", p)
            # Remove missing rasters from settings
            settings["rasters"] = {
                k: v for k, v in settings["rasters"].items() if v.exists()
            }

        if missing_cbs or not settings["rasters"]:
            print_banner("❌  Ontbrekende bestanden")
            for p in missing_cbs + ([] if settings["rasters"] else missing_rasters):
                print(f"  • {p}")
            print(textwrap.dedent("""
              Pas config.yaml aan met de juiste bestandspaden, of raadpleeg
              README.md > "How to get the data" voor downloadinstructies.
            """))
            return 1

    # ── Import pipeline modules ──────────────────────────────────────────────
    try:
        from enrich_wijken import load_wijken, compute_zonal_stats, save_output
        from utils import (
            reproject_if_needed,
            calculate_percentage_above_threshold,
            normalize_column,
        )
    except ImportError as exc:
        log.error("Kan pipeline-modules niet importeren: %s", exc)
        log.error("Installeer dependencies: pip install -r requirements.txt")
        return 2

    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        t_start = time.perf_counter()

        gdf = load_wijken(settings["cbs_path"], settings["cbs_layer"], settings["gemeente"])
        original_cols = set(gdf.columns)

        for prefix, raster_path in settings["rasters"].items():
            gdf = reproject_if_needed(gdf, raster_path)
            gdf = compute_zonal_stats(gdf, raster_path, settings["stats"], prefix)

            if settings["threshold"] is not None:
                gdf = calculate_percentage_above_threshold(
                    gdf, raster_path, settings["threshold"], prefix
                )

            if settings["normalize"] and f"{prefix}_mean" in gdf.columns:
                col = f"{prefix}_mean"
                gdf[f"{col}_norm"] = normalize_column(gdf, col)

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
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CONFIG,
        metavar="YAML",
        help=f"Config-bestand (standaard: {DEFAULT_CONFIG.name}).",
    )
    parser.add_argument(
        "--gemeente",
        type=str,
        default=None,
        metavar="NAAM",
        help="Overschrijf de gemeente uit config.yaml.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Toon DEBUG-logberichten.",
    )
    args = parser.parse_args()
    return run(args.config, args.gemeente, args.verbose)


if __name__ == "__main__":
    sys.exit(main())