import re
from datetime import date, timedelta
from pathlib import Path

import math
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(page_title="Dam Water Levels Dashboard", layout="wide")
st.title("💧 Dam Water Levels & Status Dashboard")

CSV_PATH = Path("dams_data_new.csv")

# Map messy spreadsheet headers → canonical names used in the app
RAW_TO_CANON = {
    "Height \n(ft)": "Height (ft)",
    "Completion Cost \n(million)": "Completion Cost (million)",
    "Gross Storage Capacity \n(Aft)": "Gross Storage Capacity (Aft)",
    "Live storage \n(Aft)": "Live storage (Aft)",
    "C.C.A. \n(Acres)": "C.C.A. (Acres)",
    "Capacity of Channel \n(Cfs)": "Capacity of Channel (Cfs)",
    "Length of Canal \n(ft)": "Length of Canal (ft)",
    "DSL \n(ft)": "DSL (ft)",
    "NPL \n(ft)": "NPL (ft)",
    "HFL \n(ft)": "HFL (ft)",
    "Catchment Area \n(Sq. Km)": "Catchment Area (Sq. Km)",
}

REQUIRED_COLS = [
    "Date",
    "Location",
    "Water_Level_ft",
    "Height (ft)",
    "Gross Storage Capacity (Aft)",
    "Live storage (Aft)",
    "C.C.A. (Acres)",
    "Capacity of Channel (Cfs)",
    "Length of Canal (ft)",
    "DSL (ft)",
    "NPL (ft)",
    "HFL (ft)",
    "River / Nullah",
    "Year of Completion",
    "Catchment Area (Sq. Km)",
    "Latitude",
    "Longitude",
]

# Colors for MAP (RGB)
STATUS_COLORS = {
    "Low Storage": [255, 0, 0],       # red
    "Medium Storage": [255, 140, 0],  # orange
    "High Storage": [0, 200, 0],      # green
    "Spill Watch": [255, 215, 0],
    "Spill Anytime": [255, 69, 0],
    "Spilling": [139, 0, 0],
    "Unknown": [120, 120, 120],       # grey
}
ALERT_STATUSES = {"Spill Watch", "Spill Anytime", "Spilling"}

# For table coloring (hex)
STATUS_BG = {
    "Low Storage": "#ffdddd",         # light red
    "Medium Storage": "#fff7cc",      # light yellow
    "High Storage": "#ddffdd",        # light green
    # darker, more alarming reds for spill-related statuses
    "Spill Watch": "#ff9999",
    "Spill Anytime": "#ff6666",
    "Spilling": "#cc0000",
    "Unknown": "#eeeeee",
}

# ──────────────────────── HELPERS ────────────────────────
def _clean_header(s: str) -> str:
    s = str(s).replace('"', "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_csv():
    if not CSV_PATH.exists():
        pd.DataFrame(columns=REQUIRED_COLS).to_csv(CSV_PATH, index=False)


@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    """
    Load dams_data.csv, normalize headers, fix Location spacing,
    parse dates (D/M/Y), make numeric, and forward/back-fill static columns
    (Height, DSL, NPL, Latitude, etc.) for each dam.
    """
    ensure_csv()
    df = pd.read_csv(CSV_PATH)

    # 1) Clean headers
    df.columns = [_clean_header(c) for c in df.columns]

    # 2) Map line-broken headers to canonical
    for raw, canon in RAW_TO_CANON.items():
        raw_clean = _clean_header(raw)
        if raw in df.columns and canon not in df.columns:
            df.rename(columns={raw: canon}, inplace=True)
        elif raw_clean in df.columns and canon not in df.columns:
            df.rename(columns={raw_clean: canon}, inplace=True)

    # 3) Validate columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns after normalization: {missing}")
        st.stop()

    # 4) Parse types (day/month/year in file)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date

    # Normalize Location
    df["Location"] = df["Location"].astype(str).str.strip()

    num_cols = [
        "Water_Level_ft",
        "DSL (ft)",
        "NPL (ft)",
        "HFL (ft)",
        "Height (ft)",
        "Gross Storage Capacity (Aft)",
        "Live storage (Aft)",
        "C.C.A. (Acres)",
        "Capacity of Channel (Cfs)",
        "Length of Canal (ft)",
        "Catchment Area (Sq. Km)",
        "Latitude",
        "Longitude",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5) Fill static columns per dam (forward/back-fill)
    df = df.sort_values(["Location", "Date"])
    static_cols = [
        c for c in REQUIRED_COLS if c not in ["Date", "Location", "Water_Level_ft"]
    ]
    for col in static_cols:
        df[col] = df.groupby("Location")[col].transform(lambda s: s.ffill().bfill())

    return df.reset_index(drop=True)


def compute_status_row(wl: float, npl: float, dsl: float) -> str:
    if pd.isna(wl) or pd.isna(npl) or pd.isna(dsl):
        return "Unknown"
    if wl < dsl:
        return "Low Storage"
    if wl > npl:
        return "Spilling"       # level above crest level
    if abs(wl - npl) < 1e-9:
        return "Spill Anytime"  # exactly at crest, can spill at any time
    if abs(wl - npl) <= 2:
        return "Spill Watch"    # within 2 ft below crest, close watch
    if wl - dsl <= 5:
        return "Low Storage"
    diff = npl - wl
    if diff < 5:
        return "High Storage"
    return "Medium Storage"


def with_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Status"] = df.apply(
        lambda r: compute_status_row(r["Water_Level_ft"], r["NPL (ft)"], r["DSL (ft)"]),
        axis=1,
    )
    return df


def _pct_full_row(wl, npl, dsl):
    if pd.isna(wl) or pd.isna(npl) or pd.isna(dsl) or (npl == dsl):
        return pd.NA
    return (wl - dsl) / (npl - dsl)


def _trend_label_from_slice(
    s: pd.Series,
    dates: pd.Series,
    change_thresh: float = 0.5,        # ft – min meaningful change
    recent_window_days: int = 10       # look at last N days for "recent" slope
) -> str:
    """Classify hydrograph shape in the selected range.

    Returns one of:
    'Rising', 'Falling', 'Stable',
    'Rising then Falling', 'Falling then Rising',
    'Variable', 'No Data'
    """
    s = s.dropna()
    if len(s) < 2:
        return "No Data"

    dates = pd.to_datetime(dates)
    if dates.isna().all():
        return "No Data"

    # align indexes
    s = s.reset_index(drop=True)
    dates = dates.reset_index(drop=True)

    first = float(s.iloc[0])
    last = float(s.iloc[-1])
    overall_change = last - first
    max_val = float(s.max())
    min_val = float(s.min())

    # 1) Very small variation → Stable
    if (max_val - min_val) < change_thresh:
        return "Stable"

    # 2) Recent slope (last recent_window_days)
    end_time = dates.iloc[-1]
    start_recent = end_time - pd.Timedelta(days=recent_window_days)
    mask_recent = dates >= start_recent

    if mask_recent.sum() >= 2:
        s_recent = s[mask_recent]
        d_recent = dates[mask_recent]
        days_recent = max((d_recent.iloc[-1] - d_recent.iloc[0]).days, 1)
        slope_recent = (float(s_recent.iloc[-1]) - float(s_recent.iloc[0])) / days_recent
    else:
        # fallback: use whole-period average slope
        days_all = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
        slope_recent = overall_change / days_all

    # 3) Decide pattern
    if overall_change >= change_thresh and slope_recent >= 0:
        return "Rising"

    if overall_change <= -change_thresh and slope_recent <= 0:
        return "Falling"

    if overall_change >= change_thresh and slope_recent < 0:
        # water rose significantly but is now dropping
        return "Rising then Falling"

    if overall_change <= -change_thresh and slope_recent > 0:
        # water fell significantly but is now recovering
        return "Falling then Rising"

    return "Variable"



def compute_trend_for_loc_asof(
    df_all: pd.DataFrame, loc: str, as_of: date, window: int = 7
) -> str:
    mask = (df_all["Location"] == loc) & (df_all["Date"] <= as_of)
    hist = df_all.loc[mask].sort_values("Date")
    if hist.empty:
        return "No Data"
    start_date = as_of - timedelta(days=window - 1)
    hist = hist[hist["Date"] >= start_date]
    return _trend_label_from_slice(hist["Water_Level_ft"], hist["Date"])


def upsert_reading(df: pd.DataFrame, d: date, loc: str, wl: float) -> pd.DataFrame:
    loc_norm = str(loc).strip()
    if loc_norm not in df["Location"].unique():
        st.error("Location not found in static data. Add one base row first.")
        st.stop()
    static_row = df[df["Location"] == loc_norm].iloc[0].to_dict()
    new_row = {k: static_row.get(k, None) for k in df.columns}
    new_row["Date"] = d
    new_row["Water_Level_ft"] = wl
    m = (df["Location"] == loc_norm) & (df["Date"] == d)
    if m.any():
        df.loc[m, "Water_Level_ft"] = wl
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df.sort_values(["Date", "Location"]).reset_index(drop=True)


def save_df(df: pd.DataFrame):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()


# ───────────────────────── MAP ─────────────────────────
def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def make_map(deck_df: pd.DataFrame):
    df_map = deck_df.copy()
    df_map["lat"] = df_map["Latitude"].apply(_safe_float)
    df_map["lon"] = df_map["Longitude"].apply(_safe_float)
    df_map = df_map[df_map["lat"].notnull() & df_map["lon"].notnull()]
    if df_map.empty:
        return None

    def _row_to_record(r):
        wl = _safe_float(r["Water_Level_ft"])
        npl = _safe_float(r["NPL (ft)"])
        dsl = _safe_float(r["DSL (ft)"])
        status = str(r["Status"]) if r["Status"] is not None else "Unknown"
        is_alert = status in ALERT_STATUSES
        label = ("⚠ " if is_alert else "") + (
            str(r["Location"]) if r["Location"] is not None else ""
        )

        # color: alerts forced to yellow, others use storage color
        if is_alert:
            color = [255, 255, 0]  # bright yellow balloon
        else:
            color = STATUS_COLORS.get(status, [120, 120, 120])

        # slightly larger size for alert balloons
        size = 28 if is_alert else 22

        return {
            "LocationLabel": label,
            "Status": status,
            "WL_ft": round(wl, 2) if wl is not None else None,
            "NPL_ft": round(npl, 2) if npl is not None else None,
            "DSL_ft": round(dsl, 2) if dsl is not None else None,
            "Date_str": str(r["Date"]),
            "Latitude": float(r["lat"]),
            "Longitude": float(r["lon"]),
            "color": color,
            "icon": "alert",
            "size": size,
        }

    records = [_row_to_record(r) for _, r in df_map.iterrows()]
    if not records:
        return None

    view_state = pdk.ViewState(
        latitude=sum(rec["Latitude"] for rec in records) / len(records),
        longitude=sum(rec["Longitude"] for rec in records) / len(records),
        zoom=6,
    )

    # Single IconLayer: all dams as balloons, colors by status
    icon_url = (
        "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/"
        "icon-atlas.png"
    )
    icon_mapping = {
        "alert": {
            "x": 128,
            "y": 0,
            "width": 128,
            "height": 128,
            "anchorY": 128,
            "mask": True,
        }
    }

    layer = pdk.Layer(
        "IconLayer",
        data=records,
        get_icon="icon",
        get_size="size",
        size_scale=2,  # overall scaling (keeps balloons nicely sized)
        get_position="[Longitude, Latitude]",
        get_color="color",
        pickable=True,
        icon_atlas=icon_url,
        icon_mapping=icon_mapping,
    )

    tooltip = {
        "html": "<b>{LocationLabel}</b><br/>Status: {Status}"
        "<br/>WL: {WL_ft} ft<br/>NPL: {NPL_ft} ft<br/>DSL: {DSL_ft} ft"
        "<br/>Date: {Date_str}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)


# ────────────────────── SIDEBAR ──────────────────────
with st.sidebar:
    st.header("➕ Add / Update Daily Reading")
    df_all = load_data()
    sel_loc = st.selectbox(
        "Dam (Location)", sorted(df_all["Location"].dropna().unique())
    )
    sel_date = st.date_input("Date", value=date.today())
    wl = st.number_input("Water Level (ft)", step=0.1, format="%.2f")
    if st.button("Save Reading", use_container_width=True):
        df_new = upsert_reading(df_all, sel_date, sel_loc, wl)
        save_df(df_new)
        st.success(f"Saved {sel_loc} @ {sel_date} = {wl:.2f} ft")

# ───────────────────── MAIN DASHBOARD ─────────────────────
df = with_status(load_data())

# Filters
f1, f2, f3 = st.columns([1.1, 1.2, 1])
with f1:
    if df.empty or df["Date"].isna().all():
        default_day = date.today()
    else:
        _dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        default_day = (
            _dates.dropna().max().date() if not _dates.dropna().empty else date.today()
        )
    show_date = st.date_input("Show date", value=default_day)

with f2:
    filt_locs = st.multiselect(
        "Filter dams", sorted(df["Location"].dropna().unique())
    )
with f3:
    only_latest = st.checkbox("Quick: only latest date for each dam", value=False)

view = df.copy()
if only_latest and not view.empty:
    idx = view.groupby("Location")["Date"].transform("max") == view["Date"]
    view = view[idx]
else:
    view = view[view["Date"] == show_date]
if filt_locs:
    view = view[view["Location"].isin(filt_locs)]

# Add fraction-full for KPIs
view = view.copy()
view["Frac_Full"] = (
    view.apply(
        lambda r: _pct_full_row(
            r["Water_Level_ft"], r["NPL (ft)"], r["DSL (ft)"]
        ),
        axis=1,
    ).astype("Float64")
)
view["Frac_Full_Clip"] = view["Frac_Full"].clip(lower=0, upper=1)

# Add Trend per row (last ≤7 days up to row Date)
def _trend_row(r):
    return compute_trend_for_loc_asof(df, r["Location"], r["Date"], window=7)


view["Trend"] = view.apply(_trend_row, axis=1)
ARROWS = {
    "Rising": "▲ Rising",
    "Falling": "▼ Falling",
    "Stable": "▬ Stable",
    "Rising then Falling": "▲▼ Rise then Fall",
    "Falling then Rising": "▼▲ Fall then Rise",
    "Variable": "≈ Variable",
    "No Data": "—",
}

view["TrendDisp"] = view["Trend"].map(ARROWS).fillna("—")

# ───────── KPIs IN TABS (with smaller dam names) ─────────
st.markdown("### 🔍 Overview & Spill Alerts")
tab_overview, tab_spill = st.tabs(
    ["📊 Storage Overview", "⚠️ Spill Alerts"]
)

if not view.empty:
    # Common values
    max_val = view["Frac_Full_Clip"].max()
    min_val = view["Frac_Full_Clip"].min()
    max_names = ", ".join(
        sorted(view.loc[view["Frac_Full_Clip"] == max_val, "Location"].unique())
    )
    min_names = ", ".join(
        sorted(view.loc[view["Frac_Full_Clip"] == min_val, "Location"].unique())
    )

    alert_df = view[view["Status"].isin(ALERT_STATUSES)]

    with tab_overview:
        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            st.metric("Dams shown", view["Location"].nunique())
        with c2:
            st.markdown(
                "<span style='font-weight:600; color:#1f4e79;'>Dam(s) with Max Storage</span>",
                unsafe_allow_html=True,
            )
            if max_names:
                st.markdown(
                    f"<small style='color:#1f4e79;'>{max_names}</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("—")
        with c3:
            st.markdown(
                "<span style='font-weight:600; color:#7f0000;'>Dam(s) with Lowest Storage</span>",
                unsafe_allow_html=True,
            )
            if min_names:
                st.markdown(
                    f"<small style='color:#7f0000;'>{min_names}</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("—")

    with tab_spill:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric(
                "Spill Watch / Anytime / Spilling",
                int(alert_df["Location"].nunique()),
            )
        with c2:
            if alert_df.empty:
                st.success("No dams currently in Spill Watch / Spill Anytime / Spilling.")
            else:
                alert_names = ", ".join(sorted(alert_df["Location"].unique()))
                st.markdown(
                    f"<small style='color:#b00020; font-weight:600;'>⚠ {alert_names}</small>",
                    unsafe_allow_html=True,
                )
else:
    with tab_overview:
        st.info("No data to show.")
    with tab_spill:
        st.info("No data to show.")

# Short explanation of alert statuses
st.caption(
    "⚠ **Spill Watch**: water level within about 2 ft below NPL (crest level); "
    "**Spill Anytime**: water level is at NPL and can start spilling at any moment; "
    "**Spilling**: water level is above NPL and water is overtopping the crest."
)

# ───────── Data Table (Status color-coded) + Trend ─────────
st.markdown("### 📋 Data")
st.caption(
    "🔎 **Trend** (in the table) is based on the **last 7 days** of water levels for each dam."
)

cols_show = [
    "Date",
    "Location",
    "Water_Level_ft",
    "DSL (ft)",
    "NPL (ft)",
    "Status",
    "TrendDisp",
]
df_display = (
    view[cols_show]
    .rename(columns={"TrendDisp": "Trend"})
    .sort_values(["Location"])
    .reset_index(drop=True)
)

# Format key numeric columns to 2 decimals as strings (for display only)
for col in ["Water_Level_ft", "DSL (ft)", "NPL (ft)"]:
    if col in df_display.columns:
        df_display[col] = df_display[col].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else ""
        )


def _style_status(val):
    bg = STATUS_BG.get(val, "#eeeeee")
    text_color = "white" if val in ALERT_STATUSES else "black"
    return f"background-color: {bg}; color: {text_color}; font-weight: 600;"


def _style_trend(val):
    if isinstance(val, str) and "▲" in val:
        color = "#006400"  # dark green
    elif isinstance(val, str) and "▼" in val:
        color = "#b00020"  # dark red
    else:
        color = "#555555"
    return f"color: {color}; font-weight: 600;"


try:
    styled = (
        df_display.style.applymap(_style_status, subset=["Status"]).applymap(
            _style_trend, subset=["Trend"]
        )
    )
    st.dataframe(styled, use_container_width=True)
except Exception:
    st.dataframe(df_display, use_container_width=True)

# ───── Trend Chart with Custom Date Range ─────
st.markdown("### 📈 Water Level Trend (Custom Range)")
if not df.empty:
    eligible = (
        sorted(view["Location"].unique())
        if not view.empty
        else sorted(df["Location"].unique())
    )
    dam_for_trend = st.selectbox("Select a dam for trend", eligible)

    dam_df = df[df["Location"] == dam_for_trend].dropna(subset=["Date"])
    if dam_df.empty:
        st.info("No readings available for the selected dam.")
    else:
        dam_min_date = dam_df["Date"].min()
        dam_max_date = dam_df["Date"].max()
        default_start = max(dam_min_date, dam_max_date - timedelta(days=6))

        date_range = st.date_input(
            "Select date range",
            value=(default_start, dam_max_date),
            min_value=dam_min_date,
            max_value=dam_max_date,
        )

        # date_input for ranges returns a tuple (start, end)
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            # fallback if single date is returned
            start_date = dam_min_date
            end_date = dam_max_date

        if start_date > end_date:
            st.error("Start date must be on or before end date.")
        else:
            win = dam_df[
                (dam_df["Date"] >= start_date) & (dam_df["Date"] <= end_date)
            ].sort_values("Date")

            if win.empty:
                st.info("No readings available in the selected date range.")
            else:
                trend_now = _trend_label_from_slice(
                    win["Water_Level_ft"], win["Date"]
                )
                fig = px.line(
                    win,
                    x="Date",
                    y="Water_Level_ft",
                    markers=True,
                    title=f"{dam_for_trend} • {start_date} to {end_date}",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Trend over selected range: {ARROWS.get(trend_now, trend_now)}")

# Map
st.markdown("### 🗺️ Dams on Map (colored by Status)")
deck = make_map(view)
if deck is not None:
    try:
        st.pydeck_chart(deck)
        # Legend under the map
        st.markdown(
            "****  \n"
            "🟥 Low Storage &nbsp;&nbsp;&nbsp; "
            "🟧 Medium Storage &nbsp;&nbsp;&nbsp; "
            "🟩 High Storage &nbsp;&nbsp;&nbsp; "
            "⚠️ Yellow Balloon: Spill Watch / Anytime / Spilling"
        )
    except Exception as e:
        st.error(f"Map rendering failed: {e}")
else:
    st.info("No valid coordinates available to display the map for current filter.")

# Details
st.markdown("### 🏗️ Dam Details")
c1, c2 = st.columns([1, 2])
with c1:
    pick = st.selectbox("Select a dam", sorted(df["Location"].unique()))
with c2:
    details = (
        df[df["Location"] == pick].sort_values("Date").iloc[-1]
        if (df["Location"] == pick).any()
        else None
    )
    if details is not None:
        st.write(
            {
                "Height (ft)": details.get("Height (ft)"),
                "Gross Storage (Aft)": details.get("Gross Storage Capacity (Aft)"),
                "Live Storage (Aft)": details.get("Live storage (Aft)"),
                "C.C.A. (Acres)": details.get("C.C.A. (Acres)"),
                "Capacity of Channel (Cfs)": details.get(
                    "Capacity of Channel (Cfs)"
                ),
                "Length of Canal (ft)": details.get("Length of Canal (ft)"),
                "DSL (ft)": details.get("DSL (ft)"),
                "NPL (ft)": details.get("NPL (ft)"),
                "HFL (ft)": details.get("HFL (ft)"),
                "River / Nullah": details.get("River / Nullah"),
                "Year of Completion": details.get("Year of Completion"),
                "Catchment Area (Sq. Km)": details.get(
                    "Catchment Area (Sq. Km)"
                ),
                "Latitude": details.get("Latitude"),
                "Longitude": details.get("Longitude"),
                "Latest WL (ft)": details.get("Water_Level_ft"),
                "Status": details.get("Status"),
            }
        )
    else:
        st.info("No details available.")

st.caption("Developer: **Junaid Ahmad, PhD**")
