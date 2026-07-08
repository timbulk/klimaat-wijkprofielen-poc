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
def list_gpkg_layers(gpkg_path: str) -> list[str]:
    """Return the layer names available in a GeoPackage.

    Uses fiona to read the layer list without loading any geometries.
    Returns an empty list when the file does not exist or cannot be read.
    """
    import fiona
    path = Path(gpkg_path)
    if not path.exists():
        return []
    try:
        return fiona.listlayers(str(path))
    except Exception:
        return []


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


# Qualitative palette for categorical columns (e.g. wijktype_definitief)
_CAT_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]


def make_choropleth(
    gdf,
    column: str,
    tooltip_extra=None,
    wms_overlays=None,
    wms_url=None,
):
    """Create a Folium choropleth map for *column*.

    Parameters
    ----------
    column:        Column to visualise (numeric or categorical).
    tooltip_extra: Additional columns shown in the hover tooltip.
    wms_overlays:  Dict {label: wms_layer_name} added as toggleable WMS overlays.
    wms_url:       Base WMS endpoint URL for overlays.
    """
    import folium
    from folium.plugins import Fullscreen

    gdf_wgs = gdf.to_crs("EPSG:4326")
    bounds  = gdf_wgs.total_bounds
    center  = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    Fullscreen().add_to(m)

    is_cat = gdf_wgs[column].dtype == object or str(gdf_wgs[column].dtype) == "category"
    extra      = [c for c in (tooltip_extra or []) if c in gdf_wgs.columns and c != column]
    tip_fields = [column] + extra

    if is_cat:
        cats      = sorted(gdf_wgs[column].dropna().unique().tolist())
        color_map = {c: _CAT_COLORS[i % len(_CAT_COLORS)] for i, c in enumerate(cats)}

        def _style(feature, _cm=color_map):
            return {
                "fillColor":   _cm.get(feature["properties"].get(column), "#cccccc"),
                "fillOpacity": 0.7,
                "color":       "white",
                "weight":      0.5,
            }

        folium.GeoJson(
            gdf_wgs,
            name=column,
            style_function=_style,
            tooltip=folium.GeoJsonTooltip(fields=tip_fields, aliases=tip_fields, localize=True),
        ).add_to(m)

        legend = (
            "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;background:white;"
            "padding:10px 14px;border-radius:6px;font-size:12px;max-height:220px;"
            "overflow-y:auto;box-shadow:2px 2px 6px rgba(0,0,0,.3)'>"
            f"<b>{column}</b><br>"
        )
        for cat in cats[:18]:
            legend += (
                f"<span style='background:{color_map[cat]};display:inline-block;"
                f"width:12px;height:12px;margin-right:5px;border-radius:2px'></span>{cat}<br>"
            )
        if len(cats) > 18:
            legend += f"<i>... en {len(cats)-18} meer</i>"
        legend += "</div>"
        m.get_root().html.add_child(folium.Element(legend))

    else:
        _plot_data = gdf_wgs[[column]].reset_index()
        _plot_data[column] = pd.to_numeric(_plot_data[column], errors="coerce")
        folium.Choropleth(
            geo_data=gdf_wgs.__geo_interface__,
            data=_plot_data,
            columns=["index", column],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.3,
            nan_fill_color="#cccccc",
            legend_name=column,
            name=column,
        ).add_to(m)
        folium.GeoJson(
            gdf_wgs,
            name=f"{column} tooltip",
            tooltip=folium.GeoJsonTooltip(fields=tip_fields, aliases=tip_fields, localize=True),
            style_function=lambda _: {"fillOpacity": 0, "weight": 0},
        ).add_to(m)

    if wms_overlays and wms_url:
        for label, layer_name in wms_overlays.items():
            folium.WmsTileLayer(
                url=wms_url,
                layers=layer_name,
                fmt="image/png",
                transparent=True,
                name=f"WMS: {label}",
                opacity=0.55,
                show=False,
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m



def _render_uitleg() -> None:
    """Render the explanation tab."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.path import Path as MplPath

    st.header("Hoe werkt de analyse?")
    st.markdown(
        "Deze tool verrijkt CBS-buurten met klimaatdata door per buurt "
        "**zonal statistics** te berekenen. "
        "Hieronder leggen we stap voor stap uit hoe dat werkt."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "### \U00000031\ufe0f\u20e3 CBS buurtvlakken\n\n"
            "De CBS Wijk- en Buurtenkaart bevat alle buurtgrenzen als polygonen. "
            "Elke polygoon stelt \xe9\xe9n buurt voor."
        )
        st.caption("\U0001f5fa\ufe0f Buurtpolygoon \u2014 bijv. *Woensel-Noord* in Eindhoven")
    with c2:
        st.markdown(
            "### \U00000032\ufe0f\u20e3 WMS rasterlaag\n\n"
            "De Klimaateffectatlas levert klimaatdata als een raster: een grid waarbij "
            "elke cel een meetwaarde heeft, bijv. gevoelstemperatuur in \xb0C."
        )
        st.caption("\U0001f321\ufe0f Raster \u2014 elke pixel = meetwaarde voor dat stukje grond")
    with c3:
        st.markdown(
            "### \U00000033\ufe0f\u20e3 Zonal statistics\n\n"
            "Per buurt worden alle rasterwaarden **binnen** de polygoon verzameld "
            "en samengevat: gemiddelde, maximum, standaarddeviatie, etc."
        )
        st.caption("\U0001f4ca Uitkomst \u2014 bijv. gemiddelde hitte = 34.2 \xb0C")

    st.divider()
    st.subheader("Visueel voorbeeld: hitte-eiland effect")
    st.markdown(
        "Stel: we analyseren het **hitte-eiland effect** voor een buurt in Eindhoven. "
        "Het raster heeft een resolutie van 50 m \u2014 elke pixel dekt 50\xd750 meter."
    )

    np.random.seed(42)
    G = 12
    x = np.linspace(0, 1, G); y = np.linspace(0, 1, G)
    xx, yy = np.meshgrid(x, y)
    raster = (28
        + 8 * np.exp(-((xx-0.65)**2 + (yy-0.55)**2) / 0.08)
        + 4 * np.exp(-((xx-0.30)**2 + (yy-0.40)**2) / 0.12)
        + np.random.normal(0, 0.6, (G, G)))
    raster = np.clip(raster, 26, 40)

    poly = np.array([
        [0.15,0.20],[0.55,0.10],[0.85,0.25],[0.90,0.60],
        [0.70,0.85],[0.35,0.90],[0.10,0.70],[0.12,0.40],
    ])
    path  = MplPath(poly)
    cw    = 1.0 / G
    inside = np.array([
        [path.contains_point(((c+.5)*cw,(r+.5)*cw)) for c in range(G)]
        for r in range(G)
    ])

    vals   = raster[inside]
    n_pix  = len(vals)
    mu     = vals.mean()
    mx     = vals.max()
    sd     = vals.std()

    cmap_h = LinearSegmentedColormap.from_list("h", ["#fee8c8","#fdbb84","#e34a33","#b30000"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#0e1117")

    panel_titles = [
        "\u2460 Rasterlaag (gevoelstemperatuur \xb0C)",
        "\u2461 Buurtvlak over raster",
        "\u2462 Pixels binnen de buurt",
    ]
    for ax, t in zip(axes, panel_titles):
        ax.set_facecolor("#0e1117"); ax.set_title(t, color="white", fontsize=9, pad=8)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    im = axes[0].imshow(np.flipud(raster), extent=[0,1,0,1],
                        cmap=cmap_h, vmin=26, vmax=40, origin="upper")
    cb = fig.colorbar(im, ax=axes[0], fraction=.046, pad=.04)
    cb.set_label("\xb0C", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    axes[1].imshow(np.flipud(raster), extent=[0,1,0,1],
                   cmap=cmap_h, vmin=26, vmax=40, origin="upper", alpha=0.55)
    axes[1].add_patch(MplPolygon(poly, closed=True,
                                 edgecolor="#00cfff", facecolor="none", lw=2))
    axes[1].text(0.5, 0.5, "Buurt\nWoensel-Noord",
                 ha="center", va="center", color="white", fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#00000088", ec="none"))

    grey = np.full_like(raster, 27.0)
    axes[2].imshow(np.flipud(grey), extent=[0,1,0,1],
                   cmap="Greys", vmin=24, vmax=42, origin="upper", alpha=0.25)
    axes[2].imshow(np.flipud(np.ma.masked_where(~inside, raster)), extent=[0,1,0,1],
                   cmap=cmap_h, vmin=26, vmax=40, origin="upper")
    axes[2].add_patch(MplPolygon(poly, closed=True,
                                 edgecolor="#00cfff", facecolor="none", lw=2))
    axes[2].text(0.97, 0.03,
                 f"n = {n_pix} pixels\nmean = {mu:.1f} \xb0C\nmax  = {mx:.1f} \xb0C\nstd  = {sd:.1f} \xb0C",
                 transform=axes[2].transAxes, ha="right", va="bottom",
                 color="white", fontsize=8, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#00000099", ec="#00cfff", lw=1))

    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.subheader("Voorbeeld-uitkomst in de data")
    st.markdown(
        f"Na de analyse krijgt elke buurt nieuwe kolommen in de output GeoPackage:\n\n"
        f"| Kolom | Waarde | Betekenis |\n"
        f"|---|---|---|\n"
        f"| `hitteeiland_mean` | **{mu:.1f} \xb0C** | Gemiddelde gevoelstemperatuur |\n"
        f"| `hitteeiland_max` | **{mx:.1f} \xb0C** | Warmste pixel in de buurt |\n"
        f"| `hitteeiland_std` | **{sd:.1f} \xb0C** | Spreiding \u2014 maat voor ruimtelijke ongelijkheid |\n"
        f"| `hitteeiland_count` | **{n_pix}** | Aantal pixels \u2014 proxy voor buurtoppervlak |\n"
        f"| `wijktype_definitief` | `Tuinstad middelhoogbouw` | Stedenbouwkundig type (via WFS) |\n\n"
        "> \U0001f4a1 **Tip:** Een hoge **max** bij lage **std** wijst op een lokale hitteplek "
        "(bijv. een heet parkeerterrein). Een hoge **std** duidt op een gemengde buurt "
        "met koele groenstroken \xe9n hete verharding naast elkaar."
    )

    with st.expander("\U0001f527 Technische details"):
        st.markdown(
            "**Implementatie**\n\n"
            "- Bibliotheek: [`rasterstats`](https://pythonhosted.org/rasterstats/)\n"
            "- Rasterdata: live WMS GetMap download als GeoTIFF (EPSG:28992)\n"
            "- Buffer: 500 m rondom gemeente-bbox (instelbaar)\n"
            "- Resolutie: standaard 50 m/pixel (instelbaar)\n"
            "- Pixelgewicht: pixels op de polygoonrand worden gewogen op overlap\n\n"
            "**Statistische maatstaven**\n\n"
            "| Statistiek | Wanneer nuttig? |\n"
            "|---|---|\n"
            "| `mean` | Algemeen klimaatniveau van de buurt |\n"
            "| `max` | Worst-case hotspots |\n"
            "| `std` | Interne variatie / ruimtelijke ongelijkheid |\n"
            "| `count` | Proxy voor buurtoppervlak |\n"
            "| `pct_above_X` | % buurt boven drempelwaarde (bijv. 35\xb0C) |\n\n"
            "**Wijktypen:** join via `BU_CODE` op WFS `wijktypen_buurten` "
            "van de Klimaateffectatlas."
        )


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

    cbs_full_path = PROJECT_ROOT / cbs_path_input

    # Lees beschikbare lagen uit het GeoPackage — geen handmatig typen meer
    available_layers = list_gpkg_layers(str(cbs_full_path))

    if available_layers:
        default_layer = cfg.get("cbs_layer") or "buurten_2023"
        default_layer_idx = (
            available_layers.index(default_layer)
            if default_layer in available_layers else 0
        )
        cbs_layer_input = st.selectbox(
            "Laag",
            available_layers,
            index=default_layer_idx,
            help=f"{len(available_layers)} lagen gevonden in het GeoPackage.",
        )
    else:
        # Bestand nog niet aanwezig — val terug op tekstinvoer
        cbs_layer_input = st.text_input(
            "Laagnaam (handmatig)",
            value=cfg.get("cbs_layer") or "buurten_2023",
            help="GeoPackage niet gevonden — voer de laagnaam handmatig in.",
        )
        if cbs_full_path.exists() is False:
            st.warning(f"⚠️ Bestand niet gevonden:\n`{cbs_path_input}`", icon="⚠️")

    # ── Gemeente ─────────────────────────────────────────────────────────────
    st.subheader("🏙️ Gemeente")

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

    # ── Wijktypen ─────────────────────────────────────────────────────────────
    st.subheader("🏘️ Wijktypen")
    use_wijktypen = st.checkbox(
        "Wijktype toevoegen als kolom",
        value=True,
        help=(
            "Voegt een 'wijktype' kolom toe via een ruimtelijke join met de "
            "Klimaateffectatlas WFS (laag: wijktypen_buurten). "
            "Geeft elke buurt een stedelijk typologieklasse, bijv. 'Stadscentrum'."
        ),
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


_tab_analyse, _tab_uitleg = st.tabs(["🔬 Analyse", "📖 Hoe werkt het?"])

with _tab_uitleg:
    _render_uitleg()

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
    _wt_cols    = [c for c in _added if c.startswith("wijktype")]
    _stat_cols  = [c for c in _added if not c.endswith("_norm") and "pct_above" not in c and not c.startswith("wijktype")]
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

    # Wijktype verdeling
    if "wijktype_definitief" in _prev_gdf.columns:
        with st.expander("🏘️ Wijktype verdeling", expanded=False):
            _wt_counts = _prev_gdf["wijktype_definitief"].value_counts().reset_index()
            _wt_counts.columns = ["Wijktype", "Aantal buurten"]
            st.dataframe(_wt_counts, use_container_width=True, hide_index=True)

    # Tabel
    st.subheader("📊 Statistieken per buurt")
    _name_col = next((c for c in ("buurtnaam", "BU_NAAM", "wijknaam", "WK_NAAM") if c in _prev_gdf.columns), None)
    _wt_col   = _wt_cols  # already computed above, avoids duplicate columns
    _disp_cols = ([_name_col] if _name_col else []) + _wt_col + _stat_cols + _thresh_cols + _norm_cols
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
    st.subheader("\U0001f5fa\ufe0f Kaart")
    _map_opts = _wt_col + _stat_cols + _thresh_cols + _norm_cols
    if _map_opts:
        _map_col = st.selectbox("Toon op kaart", _map_opts, index=0, key="map_col_select")
        _tooltip_extra = [c for c in ["wijktype_definitief"] if c in _prev_gdf.columns and c != _map_col]
        _prev_wms  = st.session_state.get("wms_layers", {})
        _show_wms  = False
        if _prev_wms:
            _show_wms = st.checkbox(
                "WMS-lagen als overlay tonen",
                value=False,
                help="Voegt de geanalyseerde WMS-lagen toe als transparante overlay. Gebruik de laagbeheerder rechts van de kaart om lagen aan/uit te zetten.",
                key="show_wms_overlay",
            )
        try:
            import streamlit_folium as stf
            stf.st_folium(
                make_choropleth(
                    _prev_gdf, _map_col,
                    tooltip_extra=_tooltip_extra,
                    wms_overlays=_prev_wms if _show_wms else None,
                    wms_url=WMS_URL,
                ),
                use_container_width=True, height=520,
            )
        except ImportError:
            import matplotlib.pyplot as plt
            _fig, _ax = plt.subplots(figsize=(10, 8))
            _prev_gdf.to_crs("EPSG:4326").plot(
                column=_map_col, ax=_ax, legend=True, cmap="YlOrRd",
                missing_kwds={"color": "#cccccc", "label": "geen data"},
            )
            _ax.set_title(f"{_map_col} \u2014 {_prev_gem}", fontsize=13)
            _ax.set_axis_off()
            st.pyplot(_fig)
            st.caption("\U0001f4a1 `pip install streamlit-folium folium` voor interactieve kaart")

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

    # Stap 3: Wijktypen toevoegen via WFS ruimtelijke join
    if use_wijktypen:
        progress.progress(88, text="Wijktypen ophalen via WFS…")
        status.info("🏘️ Wijktypen ophalen via WFS…")
        try:
            from wijktypen import join_wijktypen
            gdf = join_wijktypen(gdf)
            n_filled = gdf["wijktype_definitief"].notna().sum()
            status.info(f"✅ Wijktype toegewezen aan {n_filled}/{len(gdf)} buurten")
        except Exception as exc:
            st.warning(
                f"⚠️ Wijktypen niet beschikbaar: {exc}\n"
                "De overige resultaten worden wél opgeslagen.",
                icon="⚠️",
            )

    progress.progress(90, text="Resultaten opslaan…")

    # Opslaan in session_state — resultaten worden getoond bij volgende render
    slug = gemeente.lower().replace(" ", "_")
    st.session_state["gdf"]           = gdf
    st.session_state["original_cols"] = original_cols
    st.session_state["gemeente"]      = gemeente
    st.session_state["prefix"]        = [v[:20] for v in wms_layers_selected.values()] if use_wms else [raster_prefix]
    st.session_state["wms_layers"]      = wms_layers_selected if use_wms else {}
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
