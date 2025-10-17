# app.py
import re
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(page_title="Dam Water Levels Dashboard", layout="wide")
st.title("💧 Dam Water Levels & Status Dashboard")

CSV_PATH = Path("dams_data.csv")

# Your (updated) header names from the file
RAW_HEADERS = [
    "Date",
    "Location",
    "Water_Level_ft",
    'Height \n(ft)',
    'Completion Cost \n(million)',
    'Gross Storage Capacity \n(Aft)',
    'Live storage \n(Aft)',
    'C.C.A. \n(Acres)',
    'Capacity of Channel \n(Cfs)',
    'Length of Canal \n(ft)',
    'DSL \n(ft)',
    'NPL \n(ft)',
    'HFL \n(ft)',
    'River / Nullah',
    'Year of Completion',
    'Catchment Area \n(Sq. Km)',
    'Latitude',
    'Longitude',
]

# Canonical names used inside the app
CANON = {
    "Date": "Date",
    "Location": "Location",
    "Water_Level_ft": "Water_Level_ft",
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
    "River / Nullah": "River / Nullah",
    "Year of Completion": "Year of Completion",
    "Catchment Area \n(Sq. Km)": "Catchment Area (Sq. Km)",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

REQUIRED_COLS = [
    "Date", "Location", "Water_Level_ft",
    "Height (ft)", "Gross Storage Capacity (Aft)", "Live storage (Aft)",
    "C.C.A. (Acres)", "Capacity of Channel (Cfs)", "Length of Canal (ft)",
    "DSL (ft)", "NPL (ft)", "HFL (ft)",
    "River / Nullah", "Year of Completion",
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

# ──────────────────────── HELPERS ────────────────────────
def ensure_csv():
    if not CSV_PATH.exists():
        # create an empty file with canonical headers
        pd.DataFrame(columns=REQUIRED_COLS).to_csv(CSV_PATH, index=False)

def _clean_header(s: str) -> str:
    """Remove quotes, collapse whitespace/newlines."""
    s = str(s).replace('"', '')
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    ensure_csv()
    df = pd.read_csv(CSV_PATH)

    # 1) Clean whatever headers exist
    df.columns = [_clean_header(c) for c in df.columns]

    # 2) If the file still has the line-broken versions, map to canonical
    #    Try mapping from both RAW_HEADERS and already-cleaned variants.
    rename_map = {}
    for h in df.columns:
        # Try exact RAW mapping (with newlines) first
        for raw, canon in CANON.items():
            if _clean_header(raw) == h or raw == h:
                rename_map[h] = canon
                break
        # If none matched but header is already a canonical name, keep as is
        if h in CANON.values():
            rename_map[h] = h
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # 3) Validate
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns after normalization: {missing}")
        st.stop()

    # 4) Types
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
    return df

def compute_status_row(wl: float, npl: float, dsl: float) -> str:
    if pd.isna(wl) or pd.isna(npl) or pd.isna(dsl):
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

def upsert_reading(df: pd.DataFrame, d: date, loc: str, wl: float) -> pd.DataFrame:
    if loc not in df["Location"].unique():
        st.error(
            f"Location '{loc}' is not present in the CSV static data.\n"
            f"Add at least one row for this location with its static fields first."
        )
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

    try:
        df = df.sort_values(by=["Date", "Location"], kind="stable").reset_index(drop=True)
    except Exception:
        pass
    return df

def save_df(df: pd.DataFrame):
    df_out = df.copy()
    df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_out.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()

def make_map(deck_df: pd.DataFrame):
    df_map = deck_df.dropna(subset=["Latitude", "Longitude"]).copy()
    df_map["color"] = df_map["Status"].map(lambda s: STATUS_COLORS.get(s, [120, 120, 120]))
    if df_map.empty:
        return None
    view_state = pdk.ViewState(
        latitude=float(df_map["Latitude"].mean()),
        longitude=float(df_map["Longitude"].mean()),
        zoom=6
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[Longitude, Latitude]",
        get_fill_color="color",
        get_radius=8000,
        pickable=True,
    )
    tooltip = {
        "html": "<b>{Location}</b><br/>Status: {Status}<br/>WL: {Water_Level_ft} ft"
                "<br/>NPL: {NPL (ft)} ft<br/>DSL: {DSL (ft)} ft",
        "style": {"backgroundColor": "white", "color": "black"},
    }
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)

# ────────────────────── SIDEBAR (ENTRY) ──────────────────────
with st.sidebar:
    st.header("➕ Add / Update Daily Reading")
    df_all = load_data()
    loc = st.selectbox("Dam (Location)", sorted(df_all["Location"].dropna().unique()))
    d = st.date_input("Date", value=date.today())
    wl = st.number_input("Water Level (ft)", step=0.1, format="%.2f")
    if st.button("Save Reading", use_container_width=True):
        df_new = upsert_reading(df_all, d, loc, wl)
        save_df(df_new)
        st.success(f"Saved {loc} @ {d} = {wl:.2f} ft")

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

# KPIs
k1, k2, k3, k4 = st.columns(4)
if not view.empty:
    k1.metric("Dams shown", view["Location"].nunique())
    k2.metric("Max WL (ft)", f"{view['Water_Level_ft'].max():.2f}")
    k3.metric("Min WL (ft)", f"{view['Water_Level_ft'].min():.2f}")
    k4.metric("Spill Watch/Anytime/Spilling",
              int(view["Status"].isin(["Spill Watch", "Spill Anytime", "Spilling"]).sum()))
else:
    for k in (k1, k2, k3, k4):
        k.metric("-", "-")

# Table
st.markdown("### 📋 Data")
cols_show = ["Date", "Location", "Water_Level_ft", "DSL (ft)", "NPL (ft)", "Status"]
st.dataframe(view[cols_show].sort_values(["Location"]), use_container_width=True)

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
    st.pydeck_chart(deck)
else:
    st.info("No coordinates available to display the map for current filter.")

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
