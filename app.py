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

# GDAL / rasterio thread-safety: limit to single thread to prevent segfaults
import os as _os
_os.environ.setdefault("GDAL_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("GDAL_CACHEMAX", "128")  # MB, prevent memory spikes

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
    "Waterdiepte neerslag 70 mm/1 uur":   "waterdiepte_neerslag_70mm_2uur",
    "Droogte — neerslagtekort":           "droogte_r_neerslagtekort",
    "Hitte — gevoelstemperatuur dag":     "hitteeiland_r_hitte_dag",
    "Stedelijke hitte (LST)":             "hitteeiland_r_lst",
}

STATS_OPTIONS = ["mean", "max", "min", "std", "count", "sum", "median", "majority"]

# Hitte-eiland klassen: pixelwaarde (0-9) -> temperatuurverhoging t.o.v. buitengebied
HITTE_KLASSEN = {
    0: "0 – 0,5°C",
    1: "0,5 – 1,0°C",
    2: "1,0 – 1,5°C",
    3: "1,5 – 2,0°C",
    4: "2,0 – 2,5°C",
    5: "2,5 – 3,0°C",
    6: "3,0 – 3,5°C",
    7: "3,5 – 4,0°C",
    8: "4,0 – 4,5°C",
    9: "≥ 4,5°C",
}

# Waterdiepte neerslag klassen: pixelwaarde (palette-index) → waterdiepte
# Waarden 63/126/189/252 zijn palette-indices in de KEA WMS PNG (= klas 1–4)
WATERDIEPTE_KLASSEN = {
    0:   "geen inundatie",
    63:  "< 5 cm",
    126: "5 – 30 cm",
    189: "30 – 100 cm",
    252: "> 100 cm",
    253: "onbepaald",
}

# Afstand tot koelte: R-kanaalwaarde (0–255) → afstandsklasse
# De WMS levert een RGB-visualisatielaag; band 1 (rood) loopt van donker (dicht) naar licht (ver)
AFSTAND_KLASSEN = {
    0:   "< 100 m",
    64:  "100 – 300 m",
    128: "300 – 500 m",
    192: "500 – 1 000 m",
    240: "> 1 000 m",
}

# Koppeling laagnaam → klassen-dict (voor geclassificeerde én vertaalde lagen)
WMS_LAYER_KLASSEN: dict[str, dict] = {
    "hitteeiland":                     HITTE_KLASSEN,
    "waterdiepte_neerslag_140mm_2uur":  WATERDIEPTE_KLASSEN,
    "waterdiepte_neerslag_70mm_2uur":   WATERDIEPTE_KLASSEN,
    "Afstand_tot_koelte":               AFSTAND_KLASSEN,
}

# Lagen waarvan we weten dat de pixelwaarden klassen zijn (niet absolute meetwaarden)
WMS_CLASSIFIED_LAYERS = {
    "hitteeiland",
    "waterdiepte_neerslag_140mm_2uur",
    "waterdiepte_neerslag_70mm_2uur",
    "Afstand_tot_koelte",
    "sociale_kwetsbaarheid_hitte",
    "Nachthitte_WarmeAvond20gr",
    "Nachthitte_WarmeAvond24gr",
    "Nachthitte_WarmeNacht20gr",
    "Nachthitte_WarmeNacht24gr",
}

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

    # _klasse columns: ordinal strings -> warm choropleth via numeric proxy
    _is_klasse = column.endswith("_klasse")
    is_cat = (not _is_klasse) and (
        gdf_wgs[column].dtype == object or str(gdf_wgs[column].dtype) == "category"
    )
    extra      = [c for c in (tooltip_extra or []) if c in gdf_wgs.columns and c != column]
    tip_fields = [column] + extra

    # For temp_klasse: add a numeric proxy column for the choropleth color scale
    _HITTE_REV = {v: k for k, v in HITTE_KLASSEN.items()}
    if _is_klasse:
        _num_col = f"__klasse_num_{column}"
        gdf_wgs = gdf_wgs.copy()
        gdf_wgs[_num_col] = gdf_wgs[column].map(
            lambda v: _HITTE_REV.get(v, float("nan"))
        )
        # Show the label in tooltip, not the number
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
        _choro_col = _num_col if _is_klasse else column
        _choro_label = (
            f"{column} (klasse 0–9 = temperatuurverhoging)"
            if _is_klasse else column
        )
        _plot_data = gdf_wgs[[_choro_col]].reset_index()
        _plot_data[_choro_col] = pd.to_numeric(_plot_data[_choro_col], errors="coerce")
        folium.Choropleth(
            geo_data=gdf_wgs.__geo_interface__,
            data=_plot_data,
            columns=["index", _choro_col],
            key_on="feature.id",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.3,
            nan_fill_color="#cccccc",
            legend_name=_choro_label,
            name=_choro_label,
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
    st.subheader("🌡️ Hitte-eiland klassen")
    st.markdown(
        "De **hitteeiland**-laag uit de Klimaateffectatlas bevat geclassificeerde waarden (0–9). "
        "Elke klasse staat voor een temperatuurverhoging t.o.v. het buitengebied op een warme zomerdag:"
    )
    import pandas as _pd
    _interp = (["Vrijwel geen effect"] * 2 + ["Licht effect"] * 3
               + ["Sterk effect"] * 3 + ["Zeer sterk effect"] * 2)
    _klassen_df = _pd.DataFrame([
        {"Klasse": k, "Temperatuurverhoging": v, "Interpretatie": _interp[k]}
        for k, v in {
            0:"0 – 0,5°C", 1:"0,5 – 1,0°C", 2:"1,0 – 1,5°C",
            3:"1,5 – 2,0°C", 4:"2,0 – 2,5°C", 5:"2,5 – 3,0°C",
            6:"3,0 – 3,5°C", 7:"3,5 – 4,0°C", 8:"4,0 – 4,5°C",
            9:"≥ 4,5°C"
        }.items()
    ])
    st.table(_klassen_df)
    st.markdown(
        "> 💡 De tool voegt automatisch een kolom **`{prefix}_klasse`** toe "
        "met de leesbare waarde per buurt, naast de ruwe klassewaarde."
    )

    st.subheader("🌧️ Waterdiepte neerslag klassen")
    st.markdown(
        "De **waterdiepte neerslag**-lagen (70 mm/2 uur = T=100, 140 mm/2 uur = T=1000) "
        "bevatten ook geclassificeerde waarden:"
    )
    import pandas as _pd2
    _wd_df = _pd2.DataFrame([
        {"Klasse": k, "Waterdiepte": v,
         "Interpretatie": ["Geen overlast","Beperkte overlast","Matige overlast","Ernstige overlast","Onbepaald"][i]}
        for i, (k, v) in enumerate([(0,"geen inundatie"),(63,"< 5 cm"),(126,"5 – 30 cm"),(189,"30 – 100 cm"),(252,"> 100 cm")])
    ])
    st.table(_wd_df)
    st.markdown(
        "> 💡 Voor **Afstand tot koelte** levert de WMS een RGB-visualisatielaag. "
        "De tool berekent statistieken op band 1 (roodkanaal, 0–255) en vertaalt dit naar "
        "een globale afstandsklasse (< 100 m → > 1 000 m). "
        "Gebruik dit als indicatieve maat, niet als exacte afstand."
    )
    st.divider()
    st.subheader("Visueel voorbeeld: hitte-eiland effect")
    st.markdown(
        "Stel: we analyseren het **hitte-eiland effect** voor een buurt in Eindhoven. "
        "Het raster heeft een resolutie van 50 m \u2014 elke pixel dekt 50\xd750 meter."
    )

    st.markdown(
        """
<div style="background:#1a1a2e;border-radius:12px;padding:16px 20px;font-family:monospace;font-size:13px;color:#e0e0e0;border:1px solid #333">
<b style="color:#00cfff">Voorbeeld: Buurt Woensel-Noord</b><br><br>
Stel: het hitte-eiland raster heeft resolutie 50 m. Per buurt worden alle pixels <b>binnen</b> de polygoon verzameld:<br><br>
&nbsp;&nbsp;?? Warme pixels (klasse 5?9) &nbsp; ? &nbsp; temperatuurverhoging > 2,5?C<br>
&nbsp;&nbsp;?? Gematigde pixels (klasse 2?4) &nbsp; ? &nbsp; temperatuurverhoging 1?2?C<br>
&nbsp;&nbsp;? Koele pixels (klasse 0?1) &nbsp; ? &nbsp; vrijwel geen effect<br><br>
Uitkomst: <b>majority = 4</b> &nbsp;|&nbsp; <b>mean = 3.8</b> &nbsp;|&nbsp; <b>max = 7</b> &nbsp;|&nbsp; <b>count = 42 pixels</b>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Voorbeeld-uitkomst in de data")
    st.markdown(
        f"Na de analyse krijgt elke buurt nieuwe kolommen in de output GeoPackage:\n\n"
        f"| Kolom | Waarde | Betekenis |\n"
        f"|---|---|---|\n"
        f"| `hitteeiland_mean` | **34.2 \xb0C** | Gemiddelde gevoelstemperatuur |\n"
        f"| `hitteeiland_max` | **37.8 \xb0C** | Warmste pixel in de buurt |\n"
        f"| `hitteeiland_std` | **2.1 \xb0C** | Spreiding \u2014 maat voor ruimtelijke ongelijkheid |\n"
        f"| `hitteeiland_count` | **42** | Aantal pixels \u2014 proxy voor buurtoppervlak |\n"
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



# CBS column names for spatial join matching
_CBS_GEMEENTE_COLS = ("gemeentenaam", "GM_NAAM")
_CBS_BUURT_COLS   = ("BU_NAAM", "buurtnaam")
_CBS_WIJK_COLS    = ("WK_NAAM", "wijknaam")


def _normalise(s) -> str:
    """Lowercase + strip for fuzzy name matching."""
    if s is None:
        return ""
    return str(s).strip().lower()


def _join_excel_to_gdf(
    gdf: gpd.GeoDataFrame,
    excel_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    value_cols: list[str],
    overwrite: bool = False,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Join *value_cols* from *excel_df* onto *gdf* via normalised name match.

    Returns (enriched_gdf, stats_dict).
    """
    gdf = gdf.copy()

    # Build lookup: normalised name -> row index in excel_df
    excel_df = excel_df.copy()
    excel_df["__key__"] = excel_df[right_key].map(_normalise)

    # For each value column: merge
    stats = {"matched": 0, "unmatched": 0, "skipped_existing": []}

    for col in value_cols:
        target_col = col
        if target_col in gdf.columns and not overwrite:
            stats["skipped_existing"].append(col)
            continue
        lookup = excel_df.set_index("__key__")[col].to_dict()
        gdf[target_col] = gdf[left_key].map(_normalise).map(lookup)

    # Count matches (use first value_col that was actually added)
    added = [c for c in value_cols if c not in stats["skipped_existing"]]
    if added:
        stats["matched"]   = gdf[added[0]].notna().sum()
        stats["unmatched"] = gdf[added[0]].isna().sum()

    return gdf, stats


def _render_excel_join() -> None:
    """Render the Excel enrichment tab."""
    st.header("📎 Excel verrijking")
    st.markdown(
        "Koppel een eigen Excel-bestand aan de CBS-buurtdata. "
        "Vereiste: de Excel bevat een kolom met **gemeente-** of **buurt-/wijknamen** "
        "die overeenkomen met de CBS-data. Je kiest zelf welke kolommen worden toegevoegd."
    )

    # -- Stap 1: GeoPackage bron -----------------------------------------------
    st.subheader("📦 Stap 1 — Kies de CBS-data")
    _src = st.radio(
        "Bron",
        ["Gebruik resultaat van huidige analyse", "Upload een bestaand GeoPackage"],
        horizontal=True,
        key="excel_src_radio",
    )

    # Clear cached gpkg when user switches source
    if st.session_state.get("_excel_src_prev") != _src:
        st.session_state.pop("_excel_base_gdf", None)
    st.session_state["_excel_src_prev"] = _src

    _base_gdf = None

    if _src == "Gebruik resultaat van huidige analyse":
        _base_gdf = st.session_state.get("gdf")
        if _base_gdf is None:
            st.info(
                "ℹ️ Voer eerst een analyse uit via het tabblad **🔬 Analyse**.",
                icon="ℹ️",
            )
        else:
            st.success(
                f"✅ Analyse-resultaat geladen: "
                f"**{len(_base_gdf)}** rijen · {len(_base_gdf.columns)} kolommen"
            )
    else:
        _gpkg_upload = st.file_uploader(
            "Upload GeoPackage (.gpkg)",
            type=["gpkg"],
            key="excel_gpkg_upload",
        )
        if _gpkg_upload is not None:
            import tempfile as _tempfile2
            _tmp2 = Path(_tempfile2.mktemp(suffix=".gpkg"))
            _tmp2.write_bytes(_gpkg_upload.read())
            try:
                _loaded = gpd.read_file(_tmp2, engine="pyogrio")
                st.session_state["_excel_base_gdf"] = _loaded
            except Exception as _e:
                st.error(f"❌ GeoPackage kan niet worden gelezen: {_e}")
            finally:
                _tmp2.unlink(missing_ok=True)

        _base_gdf = st.session_state.get("_excel_base_gdf")
        if _base_gdf is not None:
            _cached_lbl = "✅ GeoPackage geladen" if _gpkg_upload else "💾 Eerder geladen GeoPackage actief"
            st.success(f"{_cached_lbl}: **{len(_base_gdf)}** rijen")
            if st.button("Wis GeoPackage", key="excel_gpkg_clear"):
                st.session_state.pop("_excel_base_gdf", None)
                st.rerun()

    if _base_gdf is None:
        return

    # -- Stap 2: Excel uploaden ------------------------------------------------
    st.divider()
    st.subheader("📄 Stap 2 — Upload Excel-bestand")
    _xl_file = st.file_uploader(
        "Upload Excel (.xlsx of .xls)",
        type=["xlsx", "xls"],
        key="excel_upload",
    )
    if _xl_file is not None:
        try:
            _xl_loaded = pd.read_excel(_xl_file)
            st.session_state["_excel_xl_df"] = _xl_loaded
        except Exception as _e:
            st.error(f"❌ Excel kan niet worden gelezen: {_e}")
            return

    _xl_df = st.session_state.get("_excel_xl_df")
    if _xl_df is None:
        return

    _xl_cached_lbl = "✅ Excel geladen" if _xl_file else "💾 Eerder geladen Excel actief"
    st.success(f"{_xl_cached_lbl}: **{len(_xl_df)}** rijen · **{len(_xl_df.columns)}** kolommen")
    _col_prev, _col_clear = st.columns([4, 1])
    with _col_prev:
        with st.expander("Voorbeeld Excel-data (eerste 5 rijen)"):
            st.dataframe(_xl_df.head(), use_container_width=True)
    with _col_clear:
        if st.button("Wis Excel", key="excel_xl_clear"):
            st.session_state.pop("_excel_xl_df", None)
            st.session_state.pop("_excel_result_gdf", None)
            st.rerun()

    # -- Stap 3: Kolom-koppeling -----------------------------------------------
    st.divider()
    st.subheader("🔗 Stap 3 — Koppelkolommen instellen")

    _xl_cols = list(_xl_df.columns)
    _cbs_cols = [c for c in _base_gdf.columns if c != "geometry"]

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**CBS-kolom (links)**")
        _cbs_default_candidates = [
            c for c in _cbs_cols
            if any(k in c.lower() for k in ("naam", "name", "bu_naam", "wk_naam"))
        ]
        _cbs_default = _cbs_default_candidates[0] if _cbs_default_candidates else _cbs_cols[0]
        _left_key = st.selectbox(
            "Koppelkolom uit CBS-data",
            _cbs_cols,
            index=_cbs_cols.index(_cbs_default) if _cbs_default in _cbs_cols else 0,
            key="excel_left_key",
            help="De kolom in de CBS-data waarop gekoppeld wordt (bijv. BU_NAAM of gemeentenaam).",
        )

    with col_r:
        st.markdown("**Excel-kolom (rechts)**")
        _xl_default_candidates = [
            c for c in _xl_cols
            if _normalise(c) in (_normalise(_left_key), "naam", "buurtnaam", "wijknaam", "gemeentenaam")
        ]
        _xl_default = _xl_default_candidates[0] if _xl_default_candidates else _xl_cols[0]
        _right_key = st.selectbox(
            "Koppelkolom uit Excel",
            _xl_cols,
            index=_xl_cols.index(_xl_default) if _xl_default in _xl_cols else 0,
            key="excel_right_key",
            help="De kolom in de Excel met overeenkomstige namen (bijv. buurtnaam).",
        )

    _cbs_keys  = set(_base_gdf[_left_key].map(_normalise))
    _xl_keys   = set(_xl_df[_right_key].map(_normalise))
    _matched   = _cbs_keys & _xl_keys
    _unmatched = _cbs_keys - _xl_keys
    st.caption(
        f"🔍 Match-preview: **{len(_matched)}** van {len(_cbs_keys)} CBS-rijen "
        f"gevonden in Excel · {len(_unmatched)} niet gevonden"
    )
    if _unmatched and len(_unmatched) <= 10:
        st.caption("Niet gevonden: " + ", ".join(f"`{v}`" for v in sorted(_unmatched)[:10]))

    # -- Stap 4: Kolommen selecteren -------------------------------------------
    st.divider()
    st.subheader("📊 Stap 4 — Kies te koppelen kolommen")
    _non_key_cols = [c for c in _xl_cols if c != _right_key]
    _value_cols = st.multiselect(
        "Kolommen uit Excel om toe te voegen",
        _non_key_cols,
        default=st.session_state.get("_excel_value_cols_saved", _non_key_cols[:min(5, len(_non_key_cols))]),
        key="excel_value_cols_widget",
        help="Selecteer de kolommen die je wil toevoegen aan de CBS-data.",
    )
    st.session_state["_excel_value_cols_saved"] = _value_cols

    _existing = [c for c in _value_cols if c in _base_gdf.columns]
    _overwrite = False
    if _existing:
        st.warning(
            f"⚠️ Kolommen al aanwezig in CBS-data: {', '.join(f'`{c}`' for c in _existing)}"
        )
        _overwrite = st.checkbox("Bestaande kolommen overschrijven", value=False, key="excel_overwrite")

    if not _value_cols:
        st.info("Selecteer minimaal één kolom om te koppelen.")
        return

    # -- Stap 5: Koppelen ------------------------------------------------------
    st.divider()
    if st.button("🔗 Koppelen uitvoeren", type="primary", use_container_width=True, key="excel_join_btn"):
        with st.spinner("Koppelen…"):
            _result_gdf, _stats = _join_excel_to_gdf(
                _base_gdf, _xl_df, _left_key, _right_key, _value_cols, _overwrite,
            )
        st.session_state["_excel_result_gdf"]        = _result_gdf
        st.session_state["_excel_result_stats"]       = _stats
        st.session_state["_excel_result_left_key"]    = _left_key
        st.session_state["_excel_result_value_cols"]  = _value_cols
        st.session_state["_excel_result_src"]         = _src

    # Show result (persists across reruns via session_state)
    _result_gdf = st.session_state.get("_excel_result_gdf")
    _stats      = st.session_state.get("_excel_result_stats")
    if _result_gdf is not None and _stats is not None:
        _res_left_key   = st.session_state.get("_excel_result_left_key", _left_key)
        _res_value_cols = st.session_state.get("_excel_result_value_cols", _value_cols)
        _res_src        = st.session_state.get("_excel_result_src", _src)

        st.success(
            f"✅ Gekoppeld: **{_stats['matched']}** rijen gematcht · "
            f"{_stats['unmatched']} niet gematcht"
        )
        if _stats["skipped_existing"]:
            st.info(
                "Overgeslagen (al aanwezig): "
                + ", ".join(f"`{c}`" for c in _stats["skipped_existing"])
            )

        _show_cols = (
            [_res_left_key]
            + [c for c in _res_value_cols if c not in _stats["skipped_existing"]]
        )
        _show_cols = [c for c in _show_cols if c in _result_gdf.columns]
        st.dataframe(
            _result_gdf[_show_cols].head(50),
            use_container_width=True,
            hide_index=True,
        )

        _gpkg_bytes = gdf_to_gpkg_bytes(_result_gdf)
        _fname = f"excel_verrijkt_{_res_left_key}.gpkg"
        st.download_button(
            label="⬇️ Download verrijkt GeoPackage",
            data=_gpkg_bytes,
            file_name=_fname,
            mime="application/geopackage+sqlite3",
            type="primary",
            use_container_width=True,
            key="excel_download_btn",
        )

        if _res_src == "Gebruik resultaat van huidige analyse":
            st.session_state["gdf"] = _result_gdf
            st.session_state["gpkg_bytes"] = _gpkg_bytes
            st.caption(
                "💡 Analyse-resultaat bijgewerkt met Excel-data. "
                "Bekijk de kaart en tabel in het **🔬 Analyse** tabblad."
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
        default=cfg.get("stats", ["majority", "max", "mean"]),
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


_tab_analyse, _tab_uitleg, _tab_excel = st.tabs(["🔬 Analyse", "📖 Hoe werkt het?", "📎 Excel verrijking"])

with _tab_uitleg:
    _render_uitleg()

with _tab_excel:
    _render_excel_join()

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
    _label_cols = [c for c in _added if c.endswith("_klasse")]
    _stat_cols  = [c for c in _added if not c.endswith("_norm") and "pct_above" not in c
                   and not c.startswith("wijktype") and not c.endswith("_klasse")]
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
    _disp_cols = ([_name_col] if _name_col else []) + _wt_col + _label_cols + _stat_cols + _thresh_cols + _norm_cols
    if _disp_cols:
        _df_disp = style_summary_table(_prev_gdf[_disp_cols])
        st.dataframe(
            _df_disp,
            use_container_width=True,
            height=350,
            column_config={
                **{c: st.column_config.NumberColumn(c, format="%.2f")
                   for c in _stat_cols + _thresh_cols + _norm_cols
                   if c in _df_disp.columns},
                **{c: st.column_config.TextColumn(c)
                   for c in _label_cols if c in _df_disp.columns},
            },
        )

    # Kaart
    st.subheader("\U0001f5fa\ufe0f Kaart")
    _map_opts = _wt_col + _label_cols + _stat_cols + _thresh_cols + _norm_cols
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
            st.info(
                "💡 Installeer `streamlit-folium` en `folium` voor de interactieve kaart: "
                "`pip install streamlit-folium folium`"
            )
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
                # Voor geclassificeerde lagen: zorg dat majority altijd wordt berekend
                stats_for_layer = list(selected_stats)
                if wms_layer_name in WMS_CLASSIFIED_LAYERS and "majority" not in stats_for_layer:
                    stats_for_layer = stats_for_layer + ["majority"]
                gdf = _enrich_from_raster(
                    gdf, tmp_path, prefix, stats_for_layer, threshold, normalize,
                )
                # Voeg leesbare klasse-labels toe voor geclassificeerde lagen
                if wms_layer_name in WMS_CLASSIFIED_LAYERS:
                    _klassen_dict = WMS_LAYER_KLASSEN.get(wms_layer_name, HITTE_KLASSEN)
                    # Zoek bronkolom: majority heeft voorkeur, anders majority op mean
                    src_col = f"{prefix}_majority"
                    if src_col not in gdf.columns:
                        src_col = f"{prefix}_mean"
                    if src_col in gdf.columns:
                        _klasse_keys = sorted(_klassen_dict.keys())
                        def _to_klasse(v, _kd=_klassen_dict, _keys=_klasse_keys):
                            try:
                                v_int = int(round(float(v)))
                                # Exact match first
                                if v_int in _kd:
                                    return _kd[v_int]
                                # Nearest key (for continuous-to-class mapping)
                                nearest = min(_keys, key=lambda k: abs(k - v_int))
                                return _kd[nearest]
                            except (TypeError, ValueError):
                                return None
                        _klasse_col = f"{prefix}_klasse"
                        gdf[_klasse_col] = gdf[src_col].map(_to_klasse)
                        status.info(
                            f"🏷️ Klassen toegevoegd: `{_klasse_col}`"
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


