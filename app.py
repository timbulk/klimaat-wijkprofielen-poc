"""
app.py
------
Streamlit web interface for the klimaat-wijkprofielen-poc pipeline.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
import yaml

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Klimaat Wijkprofielen",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH  = Path(__file__).parent / "config.yaml"
PROJECT_ROOT = Path(__file__).parent

WMS_URL = "https://cas.cloud.sogelink.com/public/data/org/gws/YWFMLMWERURF/kea_public/wms"

# Fallback layer list — shown when the WMS is unreachable on first load
WMS_LAYERS_FALLBACK = {
    "Hitte-eiland (gevoelstemperatuur)":  "hitteeiland_r_hitte",
    "Waterdiepte neerslag 140 mm/2 uur":  "waterdiepte_neerslag_140mm_2uur",
    "Waterdiepte neerslag 70 mm/1 uur":   "waterdiepte_neerslag_70mm_1uur",
    "Droogte — neerslagtekort":           "droogte_r_neerslagtekort",
    "Hitte — gevoelstemperatuur dag":     "hitteeiland_r_hitte_dag",
    "Stedelijke hitte (LST)":             "hitteeiland_r_lst",
}

STATS_OPTIONS = ["mean", "max", "min", "std", "count", "sum", "median"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


@st.cache_data(show_spinner=False)
def load_cbs_gemeenten(gpkg_path: str, layer: str | None) -> list[str]:
    """Return a sorted list of unique gemeente names from the CBS file."""
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer if layer else 0, engine="pyogrio")
        for col in ("GM_NAAM", "gemeentenaam"):
            if col in gdf.columns:
                return sorted(gdf[col].dropna().unique().tolist())
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False, ttl=300)
def fetch_wms_layers(wms_url: str) -> dict[str, str]:
    """Fetch available layers from the WMS GetCapabilities endpoint.

    Returns a dict of {display_label: layer_name} sorted alphabetically.
    Falls back to WMS_LAYERS_FALLBACK on connection errors so the UI
    remains usable when offline or the WMS is temporarily unavailable.

    The result is cached for 5 minutes (ttl=300) to avoid repeated
    GetCapabilities requests while the user adjusts settings.
    """
    try:
        from owslib.wms import WebMapService
        wms      = WebMapService(wms_url, version="1.1.1", timeout=15)
        layers   = {
            # Use the layer title when available, otherwise fall back to the name
            (info.title or name): name
            for name, info in wms.contents.items()
            if info.title or name  # skip anonymous entries
        }
        return dict(sorted(layers.items(), key=lambda x: x[0].lower()))
    except Exception:
        return WMS_LAYERS_FALLBACK


@st.cache_data(show_spinner=False)
def read_existing_gpkg_layers(gpkg_path: str) -> dict:
    """Read an existing output GeoPackage and return info about its contents.

    Returns a dict with:
      - "columns": list of all non-geometry column names
      - "stat_columns": columns that look like enriched stat columns (contain _)
      - "prefixes": unique prefixes derived from stat column names
      - "n_rows": number of features

    Returns an empty dict when the file does not exist or cannot be read.
    """
    path = Path(gpkg_path)
    if not path.exists():
        return {}
    try:
        gdf = gpd.read_file(path, engine="pyogrio")
        all_cols  = [c for c in gdf.columns if c != "geometry"]
        # Stat columns have pattern {prefix}_{statname}
        stat_cols = [c for c in all_cols if "_" in c and c.rsplit("_", 1)[-1]
                     in ("mean", "max", "min", "std", "count", "sum", "median",
                         "range", "majority", "minority", "variety")]
        prefixes  = sorted({c.rsplit("_", 1)[0] for c in stat_cols})
        return {
            "columns":     all_cols,
            "stat_columns": stat_cols,
            "prefixes":    prefixes,
            "n_rows":      len(gdf),
        }
    except Exception:
        return {}


def gdf_to_gpkg_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    """Serialize a GeoDataFrame to GeoPackage bytes for download."""
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    gdf.to_file(tmp_path, driver="GPKG", engine="pyogrio")
    data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return data


def style_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a rounded display copy of numeric stat columns."""
    display = df.copy()
    num_cols = display.select_dtypes("number").columns
    display[num_cols] = display[num_cols].round(2)
    return display


def make_choropleth(gdf: gpd.GeoDataFrame, column: str) -> object:
    """Create a Folium choropleth map for *column*."""
    import folium
    from folium.plugins import Fullscreen

    gdf_wgs = gdf.to_crs("EPSG:4326")
    bounds  = gdf_wgs.total_bounds  # [minx, miny, maxx, maxy]
    center  = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    Fullscreen().add_to(m)

    # Choropleth layer
    folium.Choropleth(
        geo_data=gdf_wgs.__geo_interface__,
        data=gdf_wgs[[column]].reset_index(),
        columns=["index", column],
        key_on="feature.id",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.3,
        nan_fill_color="#cccccc",
        legend_name=column,
    ).add_to(m)

    # Tooltip on hover
    folium.GeoJson(
        gdf_wgs,
        tooltip=folium.GeoJsonTooltip(
            fields=[column],
            aliases=[column],
            localize=True,
        ),
        style_function=lambda _: {"fillOpacity": 0, "weight": 0},
    ).add_to(m)

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

cfg = load_config()

with st.sidebar:
    st.image(
        "https://www.klimaateffectatlas.nl/images/KEA_logo.png",
        width=180,
    )
    st.title("Klimaat Wijkprofielen")
    st.caption("Proof of concept — CBS × Klimaateffectatlas")
    st.divider()

    # ── CBS bestand ──────────────────────────────────────────────────────────
    st.subheader("📂 CBS data")
    cbs_path_input = st.text_input(
        "GeoPackage pad",
        value=cfg.get("cbs_path", "data/raw/wijkenbuurten_2023.gpkg"),
        help="Pad naar het CBS Wijk- en Buurtenkaart GeoPackage.",
    )
    cbs_layer_input = st.text_input(
        "Laagnaam",
        value=cfg.get("cbs_layer") or "buurten_2023",
        help="Laagnaam in het GeoPackage, bijv. 'buurten_2023'.",
    )

    # ── Gemeente ─────────────────────────────────────────────────────────────
    st.subheader("🏙️ Gemeente")
    cbs_full_path = PROJECT_ROOT / cbs_path_input

    gemeenten = load_cbs_gemeenten(str(cbs_full_path), cbs_layer_input)
    default_gemeente = cfg.get("gemeente", "Eindhoven")

    if gemeenten:
        default_idx = gemeenten.index(default_gemeente) if default_gemeente in gemeenten else 0
        gemeente = st.selectbox("Kies gemeente", gemeenten, index=default_idx)
    else:
        gemeente = st.text_input(
            "Gemeente (handmatig invoeren)",
            value=default_gemeente,
            help="CBS-bestand niet gevonden — voer de naam handmatig in.",
        )
        if not cbs_full_path.exists():
            st.warning(f"⚠️ CBS-bestand niet gevonden:\n`{cbs_path_input}`")

    st.divider()

    # ── Bestaand GeoPackage inlezen ───────────────────────────────────────────
    st.subheader("📦 Bestaand GeoPackage")
    existing_gpkg = st.text_input(
        "Pad naar bestaand output GeoPackage (optioneel)",
        value="",
        placeholder="output/eindhoven_klimaat.gpkg",
        help="Lees een eerder gegenereerd GeoPackage in om te zien welke lagen er al in zitten.",
    )

    existing_info = {}
    if existing_gpkg:
        existing_info = read_existing_gpkg_layers(
            str(PROJECT_ROOT / existing_gpkg) if not Path(existing_gpkg).is_absolute()
            else existing_gpkg
        )
        if existing_info:
            st.success(
                f"✅ {existing_info['n_rows']} features · "
                f"{len(existing_info['stat_columns'])} stat-kolommen"
            )
            if existing_info["prefixes"]:
                st.caption("Aanwezige lagen: " +
                           ", ".join(f"`{p}`" for p in existing_info["prefixes"]))
            with st.expander("Alle kolommen bekijken"):
                st.write(existing_info["columns"])
        else:
            st.warning("Bestand niet gevonden of kan niet worden gelezen.")

    st.divider()

    # ── Databron ─────────────────────────────────────────────────────────────
    st.subheader("🛰️ Databron")
    source_mode = st.radio(
        "Kies databron",
        ["WMS (live download)", "Lokaal raster (.tif)"],
        index=0,
        help="WMS downloadt automatisch de juiste uitsnede voor de gemeente.",
    )

    use_wms    = source_mode == "WMS (live download)"
    wms_layer  = None
    local_raster_path = None

    if use_wms:
        with st.spinner("WMS-lagen ophalen…"):
            available_layers = fetch_wms_layers(WMS_URL)

        if available_layers is WMS_LAYERS_FALLBACK:
            st.warning("⚠️ WMS niet bereikbaar — vaste lijst wordt getoond.", icon="⚠️")

        # Pre-select the default layer from config if it exists in the list
        default_layer_name = cfg.get("wms_layer", "hitteeiland_r_hitte")
        layer_names        = list(available_layers.values())
        layer_labels       = list(available_layers.keys())
        default_idx        = (
            layer_names.index(default_layer_name)
            if default_layer_name in layer_names else 0
        )

        # Pre-selecteer de standaardlaag uit config.yaml
        default_label = layer_labels[default_idx] if layer_labels else None
        default_selection = [default_label] if default_label else []

        wms_labels = st.multiselect(
            "WMS-lagen (meerdere mogelijk)",
            layer_labels,
            default=default_selection,
            help=f"{len(available_layers)} lagen beschikbaar. Selecteer één of meer lagen — elke laag krijgt eigen kolommen in de output.",
        )
        # Maak een dict van geselecteerde {label: layer_name}
        wms_layers_selected = {label: available_layers[label] for label in wms_labels}

        if wms_layers_selected:
            # Toon per laag of de prefix al aanwezig is in een bestaand GeoPackage
            captions = []
            for lbl, lname in wms_layers_selected.items():
                prefix = lname[:20]
                already = (
                    existing_info.get("prefixes") and
                    prefix in existing_info["prefixes"]
                )
                marker = " ⚠️ _al aanwezig_" if already else ""
                captions.append(f"`{prefix}`{marker}")
            st.caption("Prefixen in output: " + "  ·  ".join(captions))
        else:
            st.warning("Selecteer minimaal één WMS-laag.", icon="⚠️")

        col1, col2 = st.columns(2)
        with col1:
            wms_resolution = st.slider(
                "Resolutie (m/px)", min_value=10, max_value=200,
                value=int(cfg.get("wms_resolution", 50)), step=10,
                help="Kleinere waarde = meer detail, langere download.",
            )
        with col2:
            wms_buffer = st.slider(
                "Buffer (m)", min_value=0, max_value=2000,
                value=int(cfg.get("wms_buffer", 500)), step=100,
                help="Buffer rondom de gemeente-bbox.",
            )
    else:
        local_raster_path = st.text_input(
            "Pad naar .tif bestand",
            value="data/raw/hitte_gevoelstemperatuur.tif",
        )
        raster_prefix = st.text_input(
            "Kolomprefix",
            value="hitte",
            help="Wordt prefix van de nieuwe kolommen, bijv. 'hitte' → 'hitte_mean'.",
        )
        wms_resolution = 50
        wms_buffer     = 500

    st.divider()

    # ── Statistieken ─────────────────────────────────────────────────────────
    st.subheader("📊 Statistieken")
    selected_stats = st.multiselect(
        "Te berekenen statistieken",
        STATS_OPTIONS,
        default=cfg.get("stats", ["mean", "max", "std", "count"]),
    )

    use_threshold = st.checkbox("Drempelwaarde berekenen", value=False)
    threshold = None
    if use_threshold:
        threshold = st.slider(
            "Drempelwaarde", min_value=0.0, max_value=100.0,
            value=float(cfg.get("threshold") or 30.0), step=0.5,
            help="% pixels boven deze waarde wordt als extra kolom toegevoegd.",
        )

    normalize = st.checkbox(
        "Normaliseer mean-kolommen (0–1)",
        value=False,
        help="Handig voor het vergelijken van lagen met verschillende eenheden.",
    )

    st.divider()

    # ── Run-knop ─────────────────────────────────────────────────────────────
    _no_layers = use_wms and not wms_layers_selected
    run_button = st.button(
        "▶️ Run Analyse",
        type="primary",
        use_container_width=True,
        disabled=(not gemeente or _no_layers),
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
# Structuur (volgorde is cruciaal voor Streamlit):
#   1. Titel
#   2. Resultaten uit session_state (altijd getoond als beschikbaar)
#   3. Guard: stop als run_button niet ingedrukt (landing of resultaten al zichtbaar)
#   4. Pipeline (alleen bereikbaar als run_button == True)
# ---------------------------------------------------------------------------

st.title("🌡️ Klimaat Wijkprofielen POC")
st.markdown(
    "Verrijk CBS buurten met klimaatdata uit de Klimaateffectatlas via **zonal statistics**."
)

# ---------------------------------------------------------------------------
# Stap 1 — Toon resultaten van een eerdere run (altijd, boven de stop-guard)
# ---------------------------------------------------------------------------
# Door de resultaten vóór de st.stop() te renderen blijven ze zichtbaar
# bij elke re-render (scrollen, sidebar-interactie, kaart-interactie).

_prev_gdf   = st.session_state.get("gdf")
_prev_cols  = st.session_state.get("original_cols", set())
_prev_gem   = st.session_state.get("gemeente", "")

if _prev_gdf is not None:
    _added      = [c for c in _prev_gdf.columns if c not in _prev_cols and c != "geometry"]
    _stat_cols  = [c for c in _added if not c.endswith("_norm") and "pct_above" not in c]
    _thresh_cols= [c for c in _added if "pct_above" in c]
    _norm_cols  = [c for c in _added if c.endswith("_norm")]

    st.divider()
    st.subheader(f"📋 Resultaten — {_prev_gem}")

    # KPI metrics
    _kpi_cols = st.columns(min(4, len(_stat_cols)) or 1)
    for _i, _col in enumerate(_stat_cols[:4]):
        with _kpi_cols[_i % len(_kpi_cols)]:
            _val  = _prev_gdf[_col].mean()
            _vmax = _prev_gdf[_col].max()
            st.metric(_col, f"{_val:.2f}" if _val is not None else "—",
                      delta=f"max {_vmax:.2f}", delta_color="off")

    # Download-knop — leest altijd uit session_state, nooit opnieuw berekend
    st.divider()
    _col_dl, _col_info = st.columns([1, 3])
    with _col_dl:
        st.download_button(
            label="⬇️ Download GeoPackage",
            data=st.session_state["gpkg_bytes"],
            file_name=st.session_state["gpkg_filename"],
            mime="application/geopackage+sqlite3",
            use_container_width=True,
            type="primary",
        )
    with _col_info:
        st.caption(
            f"**{len(_prev_gdf)}** buurten · **{len(_added)}** nieuwe kolommen · "
            f"CRS: {_prev_gdf.crs.to_string() if _prev_gdf.crs else '—'}"
        )

    # Tabel
    st.subheader("📊 Statistieken per buurt")
    _name_col = next((c for c in ("buurtnaam", "BU_NAAM", "wijknaam", "WK_NAAM") if c in _prev_gdf.columns), None)
    _disp_cols = ([_name_col] if _name_col else []) + _stat_cols + _thresh_cols + _norm_cols
    if _disp_cols:
        _df_disp = style_summary_table(_prev_gdf[_disp_cols])
        st.dataframe(
            _df_disp,
            use_container_width=True,
            height=350,
            column_config={
                c: st.column_config.NumberColumn(c, format="%.2f")
                for c in _stat_cols + _thresh_cols + _norm_cols
                if c in _df_disp.columns
            },
        )

    # Kaart
    st.subheader("🗺️ Kaart")
    _map_opts = _stat_cols + _thresh_cols + _norm_cols
    if _map_opts:
        _map_col = st.selectbox("Toon op kaart", _map_opts, index=0, key="map_col_select")
        try:
            import streamlit_folium as stf
            stf.st_folium(make_choropleth(_prev_gdf, _map_col),
                          use_container_width=True, height=500)
        except ImportError:
            import matplotlib.pyplot as plt
            _fig, _ax = plt.subplots(figsize=(10, 8))
            _prev_gdf.to_crs("EPSG:4326").plot(
                column=_map_col, ax=_ax, legend=True, cmap="YlOrRd",
                missing_kwds={"color": "#cccccc", "label": "geen data"},
            )
            _ax.set_title(f"{_map_col} — {_prev_gem}", fontsize=13)
            _ax.set_axis_off()
            st.pyplot(_fig)
            st.caption("💡 `pip install streamlit-folium folium` voor interactieve kaart")

    st.divider()
    st.caption(
        "klimaat-wijkprofielen-poc · "
        "Data: [CBS](https://www.pdok.nl) & "
        "[Klimaateffectatlas](https://www.klimaateffectatlas.nl) · "
        "Gebouwd met Streamlit"
    )

# ---------------------------------------------------------------------------
# Stap 2 — Stop hier als de Run-knop NIET ingedrukt is
# ---------------------------------------------------------------------------
# Alles hierboven (resultaten) is al gerenderd. st.stop() voorkomt dat
# de pipeline-code hieronder bij elke re-render wordt uitgevoerd.

if not run_button:
    if _prev_gdf is None:
        # Nog geen resultaten — toon landing
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Databron", "WMS" if use_wms else "Lokaal .tif")
        col_b.metric("Gemeente", gemeente or "—")
        col_c.metric("Statistieken", ", ".join(selected_stats) if selected_stats else "—")
        st.info(
            "👈 Stel de analyse in via de sidebar en klik op **Run Analyse** om te starten.",
            icon="ℹ️",
        )
    st.stop()  # stop altijd — pipeline mag nooit ongewild draaien

# ---------------------------------------------------------------------------
# Stap 3 — Pipeline (alleen bereikbaar als run_button == True)
# ---------------------------------------------------------------------------

if not selected_stats:
    st.error("Selecteer minimaal één statistiek.")
    st.stop()

progress = st.progress(0, text="Opstarten…")
status   = st.empty()

try:
    from enrich_wijken import load_wijken, compute_zonal_stats, _enrich_from_raster
    from utils import reproject_if_needed, calculate_percentage_above_threshold, normalize_column
    from wms_utils import TempRaster, download_wms_as_geotiff

    # Stap 1: CBS laden
    status.info("📂 CBS data laden…")
    progress.progress(10, text="CBS laden…")

    cbs_full = PROJECT_ROOT / cbs_path_input
    if not cbs_full.exists():
        st.error(f"CBS-bestand niet gevonden: `{cbs_path_input}`")
        st.stop()

    gdf = load_wijken(cbs_full, cbs_layer_input or None, gemeente)
    original_cols = set(gdf.columns)

    progress.progress(30, text=f"{len(gdf)} buurten geladen…")
    status.info(f"✅ {len(gdf)} buurten geladen voor **{gemeente}**")
    time.sleep(0.3)

    # Stap 2: Raster verwerken
    if use_wms:
        n_layers   = len(wms_layers_selected)
        prog_start = 40
        prog_end   = 90
        prog_step  = (prog_end - prog_start) // max(n_layers, 1)

        for idx, (wms_label, wms_layer_name) in enumerate(wms_layers_selected.items()):
            prog_now = prog_start + idx * prog_step
            status.info(
                f"🛰️ ({idx + 1}/{n_layers}) Downloaden: `{wms_layer_name}`…"
            )
            progress.progress(prog_now, text=f"WMS laag {idx + 1}/{n_layers}…")

            with TempRaster(suffix=f"_{wms_layer_name}.tif") as tmp_path:
                download_wms_as_geotiff(
                    gdf,
                    wms_url=WMS_URL,
                    layer_name=wms_layer_name,
                    resolution_m=wms_resolution,
                    buffer_m=wms_buffer,
                    output_path=tmp_path,
                )
                progress.progress(prog_now + prog_step // 2,
                                  text=f"Zonal stats laag {idx + 1}/{n_layers}…")
                status.info(f"📐 ({idx + 1}/{n_layers}) Zonal statistics: `{wms_layer_name}`…")
                # Gebruik de laagnaam als prefix (max 20 tekens)
                prefix = wms_layer_name[:20]
                gdf = _enrich_from_raster(
                    gdf, tmp_path, prefix, selected_stats, threshold, normalize,
                )
    else:
        local_path = PROJECT_ROOT / local_raster_path
        if not local_path.exists():
            st.error(f"Rasterbestand niet gevonden: `{local_raster_path}`")
            st.stop()
        progress.progress(50, text="Zonal statistics berekenen…")
        status.info("📐 Zonal statistics berekenen…")
        gdf = _enrich_from_raster(
            gdf, local_path, raster_prefix, selected_stats, threshold, normalize,
        )

    progress.progress(90, text="Resultaten opslaan…")

    # Opslaan in session_state — resultaten worden getoond bij volgende render
    slug = gemeente.lower().replace(" ", "_")
    st.session_state["gdf"]           = gdf
    st.session_state["original_cols"] = original_cols
    st.session_state["gemeente"]      = gemeente
    st.session_state["prefix"]        = [v[:20] for v in wms_layers_selected.values()] if use_wms else [raster_prefix]
    st.session_state["gpkg_bytes"]    = gdf_to_gpkg_bytes(gdf)
    st.session_state["gpkg_filename"] = f"{slug}_klimaat.gpkg"

    progress.progress(100, text="Klaar!")
    status.success(f"✅ Analyse voltooid voor **{gemeente}** — {len(gdf)} buurten")

    # Trigger een re-render zodat de resultaten-sectie bovenaan verschijnt
    time.sleep(0.8)
    st.rerun()

except Exception as exc:
    progress.empty()
    status.empty()
    st.error(f"❌ Analyse mislukt: {exc}")
    with st.expander("Foutdetails"):
        import traceback
        st.code(traceback.format_exc())
    st.stop()