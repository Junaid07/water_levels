# app.py
import re
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
import math

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(page_title="Dam Water Levels Dashboard", layout="wide")
st.title("💧 Dam Water Levels & Status Dashboard")

CSV_PATH = Path("dams_data.csv")

# Map messy spreadsheet headers → canonical names used in the app
RAW_TO_CANON = {
    'Height \n(ft)': "Height (ft)",
    'Completion Cost \n(million)': "Completion Cost (million)",
    'Gross Storage Capacity \n(Aft)': "Gross Storage Capacity (Aft)",
    'Live storage \n(Aft)': "Live storage (Aft)",
    'C.C.A. \n(Acres)': "C.C.A. (Acres)",
    'Capacity of Channel \n(Cfs)': "Capacity of Channel (Cfs)",
    'Length of Canal \n(ft)': "Length of Canal (ft)",
    'DSL \n(ft)': "DSL (ft)",
    'NPL \n(ft)': "NPL (ft)",
    'HFL \n(ft)': "HFL (ft)",
    'Catchment Area \n(Sq. Km)': "Catchment Area (Sq. Km)",
}

REQUIRED_COLS = [
    "Date", "Location", "Water_Level_ft",
    "Height (ft)", "Gross Storage Capacity (Aft)", "Live storage (Aft)",
    "C.C.A. (Acres)", "Capacity of Channel (Cfs)", "Length of Canal (ft)",
    "DSL (ft)", "NPL (ft)", "HFL (ft)", "River / Nullah", "Year of Completion",
    "Catchment Area (Sq. Km)", "Latitude", "Longitude",
]

STATUS_COLORS = {
    "Low Storage": [33, 150, 243],
    "Storage Medium": [144, 202, 249],
    "Storage High": [76, 175, 80],
    "Spill Watch": [255, 152, 0],
    "Spill Anytime": [229, 57, 53],
    "Spilling": [183, 28, 28],
    "Unknown": [120, 120, 120],
}

# For table coloring (hex)
STATUS_BG = {
    "Low Storage": "#ffdddd",        # red-ish
    "Storage Medium": "#fff7cc",     # yellow-ish
    "Storage High": "#ddffdd",       # green-ish
    "Spill Watch": "#ffe8cc",
    "Spill Anytime": "#ffd6d6",
    "Spilling": "#ffcccc",
    "Unknown": "#eeeeee",
}

# ──────────────────────── HELPERS ────────────────────────
def _clean_header(s: str) -> str:
    """Remove quotes and collapse whitespace/newlines."""
    s = str(s).replace('"', '')
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_csv():
    if not CSV_PATH.exists():
        pd.DataFrame(columns=REQUIRED_COLS).to_csv(CSV_PATH, index=False)

@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
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

    # 4) Parse types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    num_cols = [
        "Water_Level_ft", "DSL (ft)", "NPL (ft)", "HFL (ft)", "Height (ft)",
        "Gross Storage Capacity (Aft)", "Live storage (Aft)",
        "C.C.A. (Acres)", "Capacity of Channel (Cfs)",
        "Length of Canal (ft)", "Catchment Area (Sq. Km)",
        "Latitude", "Longitude",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5) Fill static columns per dam (forward/back-fill)
    static_cols = [c for c in REQUIRED_COLS if c not in ["Date", "Location", "Water_Level_ft"]]
    df[static_cols] = (
        df.sort_values(["Location", "Date"])
          .groupby("Location")[static_cols]
          .apply(lambda g: g.ffill().bfill())
          .reset_index(level=0, drop=True)
    )

    return df

def compute_status_row(wl: float, npl: float, dsl: float) -> str:
    # Precedence from your spec
    if pd.isna(wl) or pd.isna(npl) or pd.isna(dsl):
        if not pd.isna(wl) and not pd.isna(dsl) and abs(wl - dsl) <= 5:
            return "Low Storage"
        return "Unknown"
    if wl > npl:
        return "Spilling"
    if abs(wl - npl) < 1e-9:
        return "Spill Anytime"
    if abs(wl - npl) <= 2:
        return "Spill Watch"
    if abs(wl - dsl) <= 5:
        return "Low Storage"
    diff = npl - wl
    if diff < 5:
        return "Storage High"
    return "Storage Medium"

def with_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Status"] = df.apply(
        lambda r: compute_status_row(r["Water_Level_ft"], r["NPL (ft)"], r["DSL (ft)"]),
        axis=1
    )
    return df

def _pct_full_row(wl, npl, dsl):
    """Fraction full between DSL and NPL; >1 means above NPL; <0 below DSL."""
    if pd.isna(wl) or pd.isna(npl) or pd.isna(dsl) or (npl == dsl):
        return pd.NA
    return (wl - dsl) / (npl - dsl)

def upsert_reading(df: pd.DataFrame, d: date, loc: str, wl: float) -> pd.DataFrame:
    if loc not in df["Location"].unique():
        st.error("Location not found in static data. Add one base row first.")
        st.stop()

    static_row = df[df["Location"] == loc].iloc[0].to_dict()
    new_row = {k: static_row.get(k, None) for k in df.columns}
    new_row["Date"] = d
    new_row["Water_Level_ft"] = wl

    m = (df["Location"] == loc) & (df["Date"] == d)
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
    """Build a robust PyDeck map using plain Python dict records."""
    df_map = deck_df.copy()

    # Only rows with valid numeric coordinates
    df_map["lat"] = df_map["Latitude"].apply(_safe_float)
    df_map["lon"] = df_map["Longitude"].apply(_safe_float)
    df_map = df_map[df_map["lat"].notnull() & df_map["lon"].notnull()]
    if df_map.empty:
        return None

    # Prepare plain dict records (native types only)
    def _row_to_record(r):
        return {
            "Location": str(r["Location"]) if r["Location"] is not None else "",
            "Status": str(r["Status"]) if r["Status"] is not None else "Unknown",
            "WL_ft": _safe_float(r["Water_Level_ft"]),
            "NPL_ft": _safe_float(r["NPL (ft)"]),
            "DSL_ft": _safe_float(r["DSL (ft)"]),
            "Date_str": str(r["Date"]),
            "Latitude": float(r["lat"]),
            "Longitude": float(r["lon"]),
            "color": STATUS_COLORS.get(str(r["Status"]), [120, 120, 120]),
        }

    records = [_row_to_record(r) for _, r in df_map.iterrows()]
    if not records:
        return None

    view_state = pdk.ViewState(
        latitude=sum([rec["Latitude"] for rec in records]) / len(records),
        longitude=sum([rec["Longitude"] for rec in records]) / len(records),
        zoom=6,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=records,                            # ← list of dicts (fully serializable)
        get_position="[Longitude, Latitude]",
        get_fill_color="color",
        get_radius=8000,
        pickable=True,
    )

    tooltip = {
        "html": "<b>{Location}</b><br/>Status: {Status}"
                "<br/>WL: {WL_ft} ft<br/>NPL: {NPL_ft} ft<br/>DSL: {DSL_ft} ft"
                "<br/>Date: {Date_str}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)

# ────────────────────── SIDEBAR ──────────────────────
with st.sidebar:
    st.header("➕ Add / Update Daily Reading")
    df_all = load_data()
    sel_loc = st.selectbox("Dam (Location)", sorted(df_all["Location"].dropna().unique()))
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
    default_day = df["Date"].max() if not df.empty else date.today()
    show_date = st.date_input("Show date", value=default_day)
with f2:
    filt_locs = st.multiselect("Filter dams", sorted(df["Location"].dropna().unique()))
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

# Add fraction-full between DSL and NPL for ranking (clipped 0..1 for display)
view = view.copy()
view["Frac_Full"] = view.apply(
    lambda r: _pct_full_row(r["Water_Level_ft"], r["NPL (ft)"], r["DSL (ft)"]),
    axis=1
).astype("Float64")
view["Frac_Full_Clip"] = view["Frac_Full"].clip(lower=0, upper=1)

# KPIs
k1, k2, k3, k4 = st.columns(4)
if not view.empty:
    k1.metric("Dams shown", view["Location"].nunique())

    # Max / Min by fraction-full
    max_val = view["Frac_Full_Clip"].max()
    min_val = view["Frac_Full_Clip"].min()
    max_names = ", ".join(sorted(view.loc[view["Frac_Full_Clip"] == max_val, "Location"].unique()))
    min_names = ", ".join(sorted(view.loc[view["Frac_Full_Clip"] == min_val, "Location"].unique()))

    k2.metric("Dam(s) with Max Capacity", max_names if max_names else "—")
    k3.metric("Dam(s) with Lowest Capacity", min_names if min_names else "—")
    k4.metric("Spill Watch/Anytime/Spilling",
              int(view["Status"].isin(["Spill Watch", "Spill Anytime", "Spilling"]).sum()))
else:
    for k in (k1, k2, k3, k4):
        k.metric("-", "-")

# Data Table (Status color-coded)
st.markdown("### 📋 Data")
cols_show = ["Date", "Location", "Water_Level_ft", "DSL (ft)", "NPL (ft)", "Status"]

def _style_status(val):
    bg = STATUS_BG.get(val, "#eeeeee")
    return f"background-color: {bg}; color: black;"

df_display = view[cols_show].sort_values(["Location"]).reset_index(drop=True)
try:
    styled = df_display.style.applymap(_style_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True)
except Exception:
    # Fallback if Styler not supported
    st.dataframe(df_display, use_container_width=True)

# Chart
if not view.empty:
    st.markdown("### 📈 Water Levels by Dam")
    fig = px.bar(
        view.sort_values(["Location"]),
        x="Location", y="Water_Level_ft", color="Status",
        color_discrete_map={k: f"rgb({v[0]},{v[1]},{v[2]})" for k, v in STATUS_COLORS.items()},
        title=f"Water Levels on {view['Date'].max() if not only_latest else 'Latest per Dam'}"
    )
    st.plotly_chart(fig, use_container_width=True)

# Map
st.markdown("### 🗺️ Dams on Map (colored by Status)")
deck = make_map(view)
if deck is not None:
    try:
        st.pydeck_chart(deck)
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
    details = df[df["Location"] == pick].sort_values("Date").iloc[-1] \
              if (df["Location"] == pick).any() else None
    if details is not None:
        st.write({
            "Height (ft)": details.get("Height (ft)"),
            "Gross Storage (Aft)": details.get("Gross Storage Capacity (Aft)"),
            "Live Storage (Aft)": details.get("Live storage (Aft)"),
            "C.C.A. (Acres)": details.get("C.C.A. (Acres)"),
            "Capacity of Channel (Cfs)": details.get("Capacity of Channel (Cfs)"),
            "Length of Canal (ft)": details.get("Length of Canal (ft)"),
            "DSL (ft)": details.get("DSL (ft)"),
            "NPL (ft)": details.get("NPL (ft)"),
            "HFL (ft)": details.get("HFL (ft)"),
            "River / Nullah": details.get("River / Nullah"),
            "Year of Completion": details.get("Year of Completion"),
            "Catchment Area (Sq. Km)": details.get("Catchment Area (Sq. Km)"),
            "Latitude": details.get("Latitude"),
            "Longitude": details.get("Longitude"),
            "Latest WL (ft)": details.get("Water_Level_ft"),
            "Status": details.get("Status"),
        })
    else:
        st.info("No details available.")

st.caption("Developer: **Junaid Ahmad, PhD**")
