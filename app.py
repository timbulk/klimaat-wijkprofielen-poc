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

        wms_label = st.selectbox(
            "WMS-laag",
            layer_labels,
            index=default_idx,
            help=f"{len(available_layers)} lagen beschikbaar op de WMS.",
        )
        wms_layer = available_layers[wms_label]
        st.caption(f"Laagnaam: `{wms_layer}`")

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
    run_button = st.button(
        "▶️ Run Analyse",
        type="primary",
        use_container_width=True,
        disabled=not gemeente,
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🌡️ Klimaat Wijkprofielen POC")
st.markdown(
    "Verrijk CBS buurten met klimaatdata uit de Klimaateffectatlas via **zonal statistics**."
)

if not run_button:
    # Geen run gevraagd — toon landing of eerdere resultaten, maar voer
    # NOOIT de pipeline opnieuw uit (voorkomt herberekening bij scrollen,
    # sidebar-interactie of andere Streamlit re-renders).
    if st.session_state.get("gdf") is None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Databron", "WMS" if use_wms else "Lokaal .tif")
        col_b.metric("Gemeente", gemeente or "—")
        col_c.metric("Statistieken", ", ".join(selected_stats) if selected_stats else "—")
        st.info(
            "👈 Stel de analyse in via de sidebar en klik op **Run Analyse** om te starten.",
            icon="ℹ️",
        )
    # Altijd stoppen als de Run-knop niet ingedrukt is — resultaten worden
    # hieronder getoond via de aparte resultaten-sectie die session_state leest.
    st.stop()

# ---------------------------------------------------------------------------
# Run pipeline — alleen uitgevoerd als run_button == True
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

    # ── Stap 1: CBS laden ────────────────────────────────────────────────────
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

    # ── Stap 2: Raster verwerken ─────────────────────────────────────────────
    if use_wms:
        status.info(f"🛰️ WMS-laag downloaden: `{wms_layer}`…")
        progress.progress(45, text="WMS downloaden…")

        with TempRaster(suffix=f"_{wms_layer}.tif") as tmp_path:
            download_wms_as_geotiff(
                gdf,
                wms_url=WMS_URL,
                layer_name=wms_layer,
                resolution_m=wms_resolution,
                buffer_m=wms_buffer,
                output_path=tmp_path,
            )
            progress.progress(65, text="Zonal statistics berekenen…")
            status.info("📐 Zonal statistics berekenen…")

            prefix = wms_layer[:20]
            gdf = _enrich_from_raster(
                gdf, tmp_path, prefix,
                selected_stats, threshold, normalize,
            )
    else:
        local_path = PROJECT_ROOT / local_raster_path
        if not local_path.exists():
            st.error(f"Rasterbestand niet gevonden: `{local_raster_path}`")
            st.stop()

        progress.progress(50, text="Zonal statistics berekenen…")
        status.info("📐 Zonal statistics berekenen…")

        gdf = _enrich_from_raster(
            gdf, local_path, raster_prefix,
            selected_stats, threshold, normalize,
        )

    progress.progress(90, text="Resultaten verwerken…")

    # ── Resultaten opslaan in session state ──────────────────────────────────
    # Sla ook de geserialiseerde bytes op zodat de download-knop beschikbaar
    # blijft bij iedere re-render, ook nadat de progress bar is verdwenen.
    slug = gemeente.lower().replace(" ", "_")
    st.session_state["gdf"]           = gdf
    st.session_state["original_cols"] = original_cols
    st.session_state["gemeente"]      = gemeente
    st.session_state["prefix"]        = wms_layer[:20] if use_wms else raster_prefix
    st.session_state["gpkg_bytes"]    = gdf_to_gpkg_bytes(gdf)
    st.session_state["gpkg_filename"] = f"{slug}_klimaat.gpkg"

    progress.progress(100, text="Klaar!")
    status.success(f"✅ Analyse voltooid voor **{gemeente}** — {len(gdf)} buurten")

except Exception as exc:
    progress.empty()
    status.empty()
    st.error(f"❌ Analyse mislukt: {exc}")
    with st.expander("Foutdetails"):
        import traceback
        st.code(traceback.format_exc())
    st.stop()

# ---------------------------------------------------------------------------
# Results section
# ---------------------------------------------------------------------------

gdf           = st.session_state.get("gdf")
original_cols = st.session_state.get("original_cols", set())
prefix        = st.session_state.get("prefix", "")

if gdf is None:
    st.stop()

added_cols  = [c for c in gdf.columns if c not in original_cols and c != "geometry"]
stat_cols   = [c for c in added_cols if not c.endswith("_norm") and "pct_above" not in c]
thresh_cols = [c for c in added_cols if "pct_above" in c]
norm_cols   = [c for c in added_cols if c.endswith("_norm")]

st.divider()
st.subheader(f"📋 Resultaten — {gemeente}")

# ── KPI metrics ─────────────────────────────────────────────────────────────
kpi_cols = st.columns(min(4, len(stat_cols)) or 1)
for i, col in enumerate(stat_cols[:4]):
    with kpi_cols[i % len(kpi_cols)]:
        val  = gdf[col].mean()
        vmax = gdf[col].max()
        st.metric(
            label=col,
            value=f"{val:.2f}" if val is not None else "—",
            delta=f"max {vmax:.2f}",
            delta_color="off",
        )

# ── Download-knop ─────────────────────────────────────────────────────────────
# Bytes zijn al berekend en opgeslagen tijdens de run — de knop blijft
# beschikbaar bij iedere re-render zonder dat de data opnieuw geserialiseerd
# hoeft te worden.
st.divider()
col_dl, col_info = st.columns([1, 3])
with col_dl:
    st.download_button(
        label="⬇️ Download GeoPackage",
        data=st.session_state["gpkg_bytes"],
        file_name=st.session_state["gpkg_filename"],
        mime="application/geopackage+sqlite3",
        use_container_width=True,
        type="primary",
    )
with col_info:
    st.caption(
        f"**{len(gdf)}** buurten · **{len(added_cols)}** nieuwe kolommen · "
        f"CRS: {gdf.crs.to_string() if gdf.crs else '—'}"
    )

# ── Samenvatting tabel ───────────────────────────────────────────────────────
st.subheader("📊 Statistieken per buurt")

name_col = next((c for c in ("buurtnaam", "BU_NAAM", "wijknaam", "WK_NAAM") if c in gdf.columns), None)
display_cols = ([name_col] if name_col else []) + stat_cols + thresh_cols + norm_cols

if display_cols:
    df_display = style_summary_table(gdf[display_cols])
    st.dataframe(
        df_display,
        use_container_width=True,
        height=350,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.2f")
            for col in stat_cols + thresh_cols + norm_cols
            if col in df_display.columns
        },
    )

# ── Kaart ────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Kaart")

map_col_options = stat_cols + thresh_cols + norm_cols
if not map_col_options:
    st.info("Geen numerieke kolommen beschikbaar voor de kaart.")
else:
    col_map_sel, col_map_info = st.columns([2, 3])
    with col_map_sel:
        map_column = st.selectbox(
            "Toon op kaart",
            map_col_options,
            index=0,
        )

    try:
        import streamlit_folium as stf
        m = make_choropleth(gdf, map_column)
        stf.st_folium(m, use_container_width=True, height=500)
    except ImportError:
        # Fallback: static matplotlib map
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.to_crs("EPSG:4326").plot(
            column=map_column,
            ax=ax,
            legend=True,
            cmap="YlOrRd",
            missing_kwds={"color": "#cccccc", "label": "geen data"},
        )
        ax.set_title(f"{map_column} — {gemeente}", fontsize=13)
        ax.set_axis_off()
        st.pyplot(fig)
        st.caption(
            "💡 Installeer `streamlit-folium` voor een interactieve kaart: "
            "`pip install streamlit-folium folium`"
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "klimaat-wijkprofielen-poc · "
    "Data: [CBS](https://www.pdok.nl) & "
    "[Klimaateffectatlas](https://www.klimaateffectatlas.nl) · "
    "Gebouwd met Streamlit"
)