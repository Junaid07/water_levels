import pandas as pd
import streamlit as st
from datetime import date
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Water Levels", layout="wide")
CSV_PATH = Path("water_levels.csv")

# --- Load or create CSV ---
if not CSV_PATH.exists():
    pd.DataFrame(columns=["Date","Location","Water_Level_ft","Remarks"]).to_csv(CSV_PATH, index=False)

@st.cache_data(ttl=2)
def load_data():
    df = pd.read_csv(CSV_PATH)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df["Water_Level_ft"] = pd.to_numeric(df["Water_Level_ft"], errors="coerce")
    return df

def save_row(d, loc, level, remarks):
    df = load_data()
    new = pd.DataFrame([{
        "Date": pd.to_datetime(d).date().isoformat(),
        "Location": loc.strip(),
        "Water_Level_ft": float(level),
        "Remarks": remarks.strip()
    }])
    out = pd.concat([df, new], ignore_index=True)
    out.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()

st.title("💧 Daily Water Levels")

# --- Data entry (left) | View (right) ---
c1, c2 = st.columns([1,2])

with c1:
    st.subheader("Add today's reading")
    with st.form("add_form", clear_on_submit=True):
        d = st.date_input("Date", value=date.today())
        df_all = load_data()
        locations = sorted(df_all["Location"].dropna().unique()) if not df_all.empty else []
        loc = st.selectbox("Location", ["— type new —"] + locations)
        loc_new = st.text_input("New Location") if loc == "— type new —" else ""
        level = st.number_input("Water Level (ft)", step=0.1, format="%.2f")
        remarks = st.text_input("Remarks", value="")
        ok = st.form_submit_button("Save")
    if ok:
        final_loc = loc_new.strip() if loc == "— type new —" else loc
        if not final_loc:
            st.error("Please select or type a location.")
        else:
            save_row(d, final_loc, level, remarks)
            st.success(f"Saved {final_loc} @ {d}")

with c2:
    st.subheader("Data & Charts")
    df = load_data()
    if df.empty:
        st.info("No data yet. Add your first reading on the left.")
    else:
        # Filters
        f1, f2 = st.columns(2)
        with f1:
            dr = st.date_input(
                "Date range",
                value=(df["Date"].min(), df["Date"].max())
            )
        with f2:
            flocs = st.multiselect("Locations", sorted(df["Location"].unique()))
        dmin, dmax = (dr if isinstance(dr, tuple) else (dr, dr))
        view = df[(df["Date"]>=dmin) & (df["Date"]<=dmax)]
        if flocs:
            view = view[view["Location"].isin(flocs)]

        st.dataframe(view.sort_values(["Date","Location"]), use_container_width=True)

        if not view.empty:
            st.plotly_chart(
                px.line(view.sort_values(["Location","Date"]),
                        x="Date", y="Water_Level_ft", color="Location", markers=True,
                        title="Water level by day"),
                use_container_width=True
            )
