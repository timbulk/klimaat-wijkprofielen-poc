#!/usr/bin/env python3
"""
test_run.py
-----------
Quick smoke test / demo run for the enrich_wijken pipeline.

Before running, fill in the two FILE PATH SECTIONS below with the
actual locations of your downloaded data files.  See README.md for
instructions on where to obtain them.

Run from the project root:
    python scripts/test_run.py

Or with verbose logging:
    python scripts/test_run.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

# Make sure utils and enrich_wijken are importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

# ===========================================================================
# ✏️  FILL IN YOUR FILE PATHS HERE
# ===========================================================================

#: Path to the CBS Wijk- en Buurtenkaart GeoPackage.
#: Download from: https://www.pdok.nl/downloads/-/article/cbs-wijk-en-buurtkaart
#: Example: "data/raw/wijkenbuurten_2023.gpkg"
CBS_GPKG = Path("data/raw/wijkenbuurten_2023.gpkg")

#: Layer name inside the GeoPackage.
#: Common values: "buurten_2023", "wijken_2023"
#: Set to None to use the first available layer.
CBS_LAYER = "buurten_2023"

#: Municipality to use for the test run.
#: Choose a medium-sized city to keep processing fast.
#: Alternatives: "Utrecht", "Haarlem", "Tilburg", "Breda"
GEMEENTE = "Eindhoven"

#: Raster 1 — Heat stress (gevoelstemperatuur).
#: Download from Klimaateffectatlas > Hitte > Gevoelstemperatuur
#: https://www.klimaateffectatlas.nl
RASTER_HITTE = Path("data/raw/hitte_gevoelstemperatuur.tif")

#: Raster 2 — Drought / rainfall deficit (neerslagtekort).
#: Download from Klimaateffectatlas > Droogte > Neerslagtekort
#: Set to None to skip this raster in the test run.
RASTER_DROOGTE = Path("data/raw/droogte_neerslagtekort.tif")

#: Output path for the enriched GeoPackage.
OUTPUT_PATH = Path("output/test_eindhoven_klimaat.gpkg")

#: Threshold value for percentage-above calculation (e.g. 30 °C for heat stress).
THRESHOLD = 30.0

# ===========================================================================
# End of configuration
# ===========================================================================


def check_files() -> tuple[bool, list[str]]:
    """Verify that the required input files exist.

    Returns
    -------
    (ok, missing_paths) — ok is True when all required files are present.
    """
    required = [CBS_GPKG, RASTER_HITTE]
    missing = [str(p) for p in required if not p.exists()]
    return len(missing) == 0, missing


def print_banner(title: str) -> None:
    """Print a simple section banner to stdout."""
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_summary(gdf, original_cols: set[str], elapsed: float) -> None:
    """Print a human-readable summary of the enrichment results.

    Parameters
    ----------
    gdf:           Enriched GeoDataFrame.
    original_cols: Column names present before enrichment.
    elapsed:       Wall-clock time in seconds.
    """
    added_cols = [c for c in gdf.columns if c not in original_cols and c != "geometry"]
    stat_cols   = [c for c in added_cols if not c.endswith("_norm") and not c.startswith("pct")]
    thresh_cols = [c for c in added_cols if "pct_above" in c]
    norm_cols   = [c for c in added_cols if c.endswith("_norm")]

    print_banner("Resultaten samenvatting")
    print(f"  Gemeente           : {GEMEENTE}")
    print(f"  Aantal buurten     : {len(gdf)}")
    print(f"  Verwerkingstijd    : {elapsed:.1f} seconden")
    print(f"  Uitvoerbestand     : {OUTPUT_PATH}")
    print()

    if stat_cols:
        print(f"  Zonal stat-kolommen ({len(stat_cols)}):")
        for col in stat_cols:
            non_null = gdf[col].notna().sum()
            val_min  = gdf[col].min()
            val_max  = gdf[col].max()
            val_mean = gdf[col].mean()
            print(f"    {col:<35}  n={non_null:>3}  "
                  f"min={val_min:>8.2f}  max={val_max:>8.2f}  gem={val_mean:>8.2f}")

    if thresh_cols:
        print(f"\n  Drempelwaarde-kolommen (>{THRESHOLD}) ({len(thresh_cols)}):")
        for col in thresh_cols:
            non_null = gdf[col].notna().sum()
            val_mean = gdf[col].mean()
            print(f"    {col:<35}  n={non_null:>3}  gemiddeld {val_mean:.1f}% boven drempel")

    if norm_cols:
        print(f"\n  Genormaliseerde kolommen ({len(norm_cols)}):")
        for col in norm_cols:
            print(f"    {col}")

    print()


def run(verbose: bool = False) -> int:
    """Execute the full test pipeline.

    Parameters
    ----------
    verbose: Enable DEBUG-level logging.

    Returns
    -------
    Exit code: 0 on success, 1 on missing files, 2 on processing error.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print_banner("klimaat-wijkprofielen-poc — test run")
    print(f"  Gemeente  : {GEMEENTE}")
    print(f"  CBS laag  : {CBS_LAYER or 'eerste laag'}")
    print(f"  Raster 1  : {RASTER_HITTE}")
    rasters_used = [RASTER_HITTE]
    if RASTER_DROOGTE and RASTER_DROOGTE.exists():
        print(f"  Raster 2  : {RASTER_DROOGTE}")
        rasters_used.append(RASTER_DROOGTE)
    elif RASTER_DROOGTE:
        print(f"  Raster 2  : {RASTER_DROOGTE}  ⚠️  niet gevonden — wordt overgeslagen")
    print(f"  Drempel   : {THRESHOLD}")
    print(f"  Output    : {OUTPUT_PATH}")

    # ── Check required files ────────────────────────────────────────────────
    ok, missing = check_files()
    if not ok:
        print_banner("❌  Ontbrekende bestanden")
        for path in missing:
            print(f"  • {path}")
        print(textwrap.dedent("""
          Vul de bestandspaden in bovenaan dit script (test_run.py),
          of raadpleeg README.md > "How to get the data" voor downloadinstructies.
        """))
        return 1

    # ── Import pipeline modules ─────────────────────────────────────────────
    try:
        from enrich_wijken import (
            load_wijken,
            compute_zonal_stats,
            save_output,
        )
        from utils import (
            reproject_if_needed,
            calculate_percentage_above_threshold,
            normalize_column,
        )
    except ImportError as exc:
        log.error("Kan pipeline-modules niet importeren: %s", exc)
        log.error("Zorg dat requirements.txt geïnstalleerd is: pip install -r requirements.txt")
        return 2

    # ── Run pipeline ────────────────────────────────────────────────────────
    try:
        t_start = time.perf_counter()

        # 1. Load CBS polygons
        gdf = load_wijken(CBS_GPKG, CBS_LAYER, GEMEENTE)
        original_cols = set(gdf.columns)

        # 2. Process each raster
        for raster_path in rasters_used:
            prefix = raster_path.stem.split("_")[0]  # e.g. "hitte", "droogte"

            # Align CRS
            gdf = reproject_if_needed(gdf, raster_path)

            # Standard zonal statistics
            gdf = compute_zonal_stats(
                gdf,
                raster_path,
                stats=["mean", "max", "std", "count"],
                prefix=prefix,
            )

            # Percentage above threshold
            gdf = calculate_percentage_above_threshold(
                gdf, raster_path, THRESHOLD, prefix
            )

            # Normalise the mean column (useful for composite scores)
            mean_col = f"{prefix}_mean"
            if mean_col in gdf.columns:
                gdf[f"{mean_col}_norm"] = normalize_column(gdf, mean_col)

        # 3. Save output
        save_output(gdf, OUTPUT_PATH)

        elapsed = time.perf_counter() - t_start

        # 4. Print summary
        print_summary(gdf, original_cols, elapsed)

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
        "--verbose", "-v",
        action="store_true",
        help="Toon DEBUG-logberichten.",
    )
    args = parser.parse_args()
    return run(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())