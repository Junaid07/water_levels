# app.py
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk
from datetime import date, datetime
from pathlib import Path

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Dam Water Levels Dashboard", layout="wide")
st.title("💧 Dam Water Levels & Status Dashboard")

CSV_PATH = Path("dams_data.csv")  # your master CSV with static fields + Date/Water_Level_ft rows

REQUIRED_COLS = [
    "Date", "Location", "Water_Level_ft",
    "Height (ft)", "Gross Storage Capacity (Aft)", "Live storage (Aft)",
    "C.C.A. (Acres)", "Capacity of Channel (Cfs)", "Length of Canal (ft)",
    "DSL (ft)", "NPL (ft)", "HFL (ft)",
    "River / Nullah", "Catchment Area (Sq. Km)",
    "Latitude", "Longitude"
]

STATUS_COLORS = {
    "Low Storage": [33, 150, 243],      # blue
    "Storage Medium": [144, 202, 249],  # light blue
    "Storage High": [76, 175, 80],      # green
    "Spill Watch": [255, 152, 0],       # orange
    "Spill Anytime": [229, 57, 53],     # red
    "Spilling": [183, 28, 28],          # dark red
}

# ------------------ HELPERS ------------------
def ensure_csv():
    if not CSV_PATH.exists():
        # make an empty file with headers if missing
        pd.DataFrame(columns=REQUIRED_COLS).to_csv(CSV_PATH, index=False)

@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    ensure_csv()
    df = pd.read_csv(CSV_PATH)
    # Normalize column names (trim whitespace/newlines)
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    # Ensure required columns exist (in case of spelling)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()
    # Convert types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Water_Level_ft"] = pd.to_numeric(df["Water_Level_ft"], errors="coerce")
    for c in ["DSL (ft)", "NPL (ft)", "HFL (ft)", "Height (ft)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Latitude" in df.columns:
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    if "Longitude" in df.columns:
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df

def compute_status_row(wl: float, npl: float, dsl: float) -> str:
    """
    Rules (precedence handled top → bottom):
    - if WL > NPL → "Spilling"
    - if WL == NPL → "Spill Anytime"
    - if WL within 2 ft of NPL → "Spill Watch"
    - if WL within 5 ft of DSL → "Low Storage"
    - else compute diff = NPL - WL:
        - diff < 5 → "Storage High"
        - diff >= 5 → "Storage Medium"
    """
    if pd.isna(wl) or pd.isna(npl) or pd.isna(dsl):
        return "Unknown"

    if wl > npl:
        return "Spilling"
    if abs(wl - npl) < 1e-9:  # equal within floating precision
        return "Spill Anytime"
    if abs(wl - npl) <= 2:
        return "Spill Watch"
    if abs(wl - dsl) <= 5:
        return "Low Storage"

    diff = npl - wl
    if diff < 5:
        return "Storage High"
    return "Storage Medium"

def add_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Status"] = df.apply(
        lambda r: compute_status_row(r["Water_Level_ft"], r["NPL (ft)"], r["DSL (ft)"]),
        axis=1
    )
    return df

def upsert_reading(df: pd.DataFrame, d: date, loc: str, wl: float) -> pd.DataFrame:
    """
    Upsert a daily reading for (Date, Location).
    Keeps the static fields from the first row found for that Location.
    If no static row exists for location, raises a user error.
    """
    if loc not in df["Location"].unique():
        st.error(f"Location '{loc}' is not present in the CSV static data. "
                 f"Add at least one row for this location with its static fields first.")
        st.stop()

    # any one row for static fields
    static_row = df[df["Location"] == loc].iloc[0].to_dict()

    # build the new row using static fields + new date/wl
    new_row = {k: static_row.get(k, None) for k in df.columns}
    new_row["Date"] = d
    new_row["Water_Level_ft"] = wl

    # check if an entry already exists for (d, loc)
    m = (df["Location"] == loc) & (df["Date"] == d)
    if m.any():
        # update WL only
        df.loc[m, "Water_Level_ft"] = wl
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Sort for cleanliness
    try:
        df = df.sort_values(by=["Date", "Location"], kind="stable").reset_index(drop=True)
    except Exception:
        pass
    return df

def save_df(df: pd.DataFrame):
    # Write as ISO dates
    df_out = df.copy()
    df_out["Date"] = pd.to_datetime(df_out["Date"]).dt.strftime("%Y-%m-%d")
    df_out.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()

def colored_map_layer(df_map: pd.DataFrame) -> pdk.Deck:
    # convert status to colors
    df_map = df_map.copy()
    df_map["color"] = df_map["Status"].map(lambda s: STATUS_COLORS.get(s, [120, 120, 120]))
    view_state = pdk.ViewState(
        latitude=float(df_map["Latitude"].mean()) if len(df_map) else 30.0,
        longitude=float(df_map["Longitude"].mean()) if len(df_map) else 70.0,
        zoom=6
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[Longitude, Latitude]",
        get_fill_color="color",
        get_radius=8000,  # meters
        pickable=True
    )
    tooltip = {
        "html": "<b>{Location}</b><br/>Status: {Status}<br/>WL: {Water_Level_ft} ft<br/>NPL: {NPL (ft)} ft<br/>DSL: {DSL (ft)} ft",
        "style": {"backgroundColor": "white", "color": "black"}
    }
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)

# ------------------ SIDEBAR: DATA ENTRY ------------------
with st.sidebar:
    st.header("➕ Add / Update Daily Reading")

    df_all = load_data()
    existing_locations = sorted(df_all["Location"].dropna().unique())
    sel_loc = st.selectbox("Dam (Location)", existing_locations)

    sel_date = st.date_input("Date", value=date.today())
    wl = st.number_input("Water Level (ft)", step=0.1, format="%.2f")

    if st.button("Save Reading", use_container_width=True):
        df_new = upsert_reading(df_all, sel_date, sel_loc, wl)
        save_df(df_new)
        st.success(f"Saved {sel_loc} @ {sel_date} = {wl:.2f} ft")

# ------------------ MAIN DASHBOARD ------------------
df = add_status(load_data())

# Filters
fl1, fl2, fl3 = st.columns([1.1, 1.2, 1])
with fl1:
    default_day = df["Date"].max() if not df.empty else date.today()
    filt_date = st.date_input("Show date", value=default_day)
with fl2:
    filt_locs = st.multiselect("Filter dams", sorted(df["Location"].dropna().unique()))
with fl3:
    only_latest = st.checkbox("Quick: only latest date for each dam", value=False)

view = df.copy()
if only_latest and not view.empty:
    # keep latest entry per dam
    idx = view.groupby("Location")["Date"].transform("max") == view["Date"]
    view = view[idx]
else:
    view = view[view["Date"] == filt_date]

if filt_locs:
    view = view[view["Location"].isin(filt_locs)]

# KPI row
k1, k2, k3, k4 = st.columns(4)
if not view.empty:
    k1.metric("Dams shown", view["Location"].nunique())
    k2.metric("Max WL (ft)", f"{view['Water_Level_ft'].max():.2f}")
    k3.metric("Min WL (ft)", f"{view['Water_Level_ft'].min():.2f}")
    k4.metric("Spill Watch/Anytime/Spilling",
              f"{(view['Status'].isin(['Spill Watch','Spill Anytime','Spilling']).sum())}")
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
map_df = view.dropna(subset=["Latitude", "Longitude"]).copy()
if not map_df.empty:
    deck = colored_map_layer(map_df)
    st.pydeck_chart(deck)
else:
    st.info("No coordinates available to display the map for current filter.")

# Dam details panel
st.markdown("### 🏗️ Dam Details")
det_col1, det_col2 = st.columns([1, 2])
with det_col1:
    dam_for_details = st.selectbox("Select a dam", sorted(df["Location"].unique()))
with det_col2:
    details = df[df["Location"] == dam_for_details].sort_values("Date").iloc[-1] if (df["Location"] == dam_for_details).any() else None
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
            "Catchment Area (Sq. Km)": details.get("Catchment Area (Sq. Km)"),
            "Latitude": details.get("Latitude"),
            "Longitude": details.get("Longitude"),
            "Latest WL (ft)": details.get("Water_Level_ft"),
            "Status": details.get("Status"),
        })
    else:
        st.info("No details available.")
        
st.caption("Developer: **Junaid Ahmad, PhD**")
