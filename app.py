import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import glob
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- 1. CONFIG & INSTITUTIONAL STYLING ---
st.set_page_config(layout="wide", page_title="Tmod Market Scout")

# Custom CSS for a tighter, professional look
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .stMetric { background-color: #F8F9FA; border: 1px solid #E9ECEF; padding: 10px; border-radius: 5px; }
        [data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #1F2937; }
        [data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #6B7280; }
        h1, h2, h3 { color: #111827; }
    </style>
""", unsafe_allow_html=True)

# --- 2. PROJECT MANAGER (SAVED SITES) ---
PROJECTS_FILE = "tmod_projects.csv"

def load_projects():
    if os.path.exists(PROJECTS_FILE):
        return pd.read_csv(PROJECTS_FILE)
    return pd.DataFrame(columns=["Name", "Address", "Target_SF"])

def save_project(name, addr, sf):
    df = load_projects()
    # Remove if exists to overwrite
    df = df[df["Name"] != name]
    new_row = pd.DataFrame([{"Name": name, "Address": addr, "Target_SF": sf}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(PROJECTS_FILE, index=False)
    return df

# --- 3. ROBUST DATA ENGINE ---
@st.cache_data
def load_data():
    # Load Redfin (Sales)
    redfin_files = [f for f in glob.glob("*.csv") if "Costar" not in f and "requirements" not in f and "tmod" not in f]
    costar_files = glob.glob("*Costar*.csv")
    
    sales_df = None
    rent_df = None

    # SALES DATA CLEANING
    if redfin_files:
        try:
            df = pd.concat([pd.read_csv(f) for f in redfin_files], ignore_index=True)
            df.columns = [c.upper() for c in df.columns] # Force Uppercase
            
            # Map columns explicitly
            df = df.rename(columns={
                'PRICE': 'Price', 'SQUARE FEET': 'SqFt', 
                'LATITUDE': 'Lat', 'LONGITUDE': 'Lon', 'YEAR BUILT': 'YearBuilt',
                'ADDRESS': 'Address', 'PROPERTY TYPE': 'PropType'
            })
            
            # Numeric conversion & Drop Junk
            cols = ['Price', 'SqFt', 'Lat', 'Lon', 'YearBuilt']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # Critical: Drop rows needed for valuation
            df = df.dropna(subset=['Price', 'SqFt', 'Lat', 'Lon'])
            
            # Fill YearBuilt 0 for visualization safety
            if 'YearBuilt' in df.columns:
                df['YearBuilt'] = df['YearBuilt'].fillna(0).astype(int)
                
            sales_df = df
        except Exception as e:
            st.error(f"Error loading Sales Data: {e}")

    # RENT DATA CLEANING
    if costar_files:
        try:
            df = pd.read_csv(costar_files[0])
            # CoStar columns are usually standard, but let's be safe
            df = df.rename(columns={'Latitude': 'Lat', 'Longitude': 'Lon'})
            rent_df = df
        except Exception:
            pass

    return sales_df, rent_df

sales_master, rent_master = load_data()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Project Manager")
    
    # Project Selector
    projects_df = load_projects()
    options = ["New Project"] + projects_df["Name"].tolist()
    selection = st.selectbox("Select Site", options)
    
    # Set Defaults
    p_name, p_addr, p_sf = "", "Boise, ID", 850
    if selection != "New Project":
        row = projects_df[projects_df["Name"] == selection].iloc[0]
        p_name = row["Name"]
        p_addr = row["Address"]
        p_sf = int(row["Target_SF"])

    # Inputs
    name_input = st.text_input("Project Name", value=p_name, placeholder="e.g. Tmod Downtown")
    address = st.text_input("Address", value=p_addr)
    target_sf = st.number_input("Target Avg Unit SF", value=p_sf, step=50)
    
    # Save Button
    if st.button("💾 Save Project", use_container_width=True):
        if name_input:
            save_project(name_input, address, target_sf)
            st.toast(f"Saved {name_input}!", icon="✅")
            st.rerun()

    st.divider()
    st.header("⚙️ Valuation Settings")
    radius = st.slider("Radius (Miles)", 0.5, 5.0, 3.0, 0.5)
    premium = st.slider("New Construction Premium", 0, 30, 15, format="%d%%") / 100

# --- 5. LOGIC & VALUATION ---
if sales_master is None:
    st.warning("⚠️ No data loaded. Please upload CSVs to GitHub.")
    st.stop()

# Geocoding
geolocator = Nominatim(user_agent="tmod_pro_analyst")
try:
    loc = geolocator.geocode(address)
except:
    st.error("⚠️ Geocoding service unavailable. Try a simpler address.")
    st.stop()

if loc:
    # Filter Functions
    def filter_radius(df, lat, lon, r_miles):
        R = 3958.8 
        dlat, dlon = np.radians(df['Lat'] - lat), np.radians(df['Lon'] - lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(df['Lat'])) * np.sin(dlon/2)**2
        df['dist'] = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))
        return df[df['dist'] <= r_miles].copy()

    s_comps = filter_radius(sales_master, loc.latitude, loc.longitude, radius)
    r_comps = pd.DataFrame()
    if rent_master is not None:
        r_comps = filter_radius(rent_master, loc.latitude, loc.longitude, radius)

    # --- VALUATION MATH ---
    est_price = 0
    if not s_comps.empty:
        # Linear Regression (Slope * Target_SF + Intercept)
        m, b = np.polyfit(s_comps['SqFt'], s_comps['Price'], 1)
        est_price = ((m * target_sf) + b) * (1 + premium)
    
    est_rent = 0
    if not r_comps.empty and 'One Bedroom Asking Rent/Unit' in r_comps.columns:
        # Rent per SF Logic (simplified for robustness)
        # Using 1-Bed columns as the anchor
        valid_rents = r_comps.dropna(subset=['One Bedroom Asking Rent/Unit', 'One Bedroom Avg SF'])
        if not valid_rents.empty:
            avg_rent_sf = (valid_rents['One Bedroom Asking Rent/Unit'] / valid_rents['One Bedroom Avg SF']).median()
            est_rent = (avg_rent_sf * target_sf) * (1 + premium)

    # --- DASHBOARD UI ---
    st.subheader(f"Project Analysis: {name_input if name_input else address}")
    
    # 1. HERO METRICS ROW
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Est. Sale Value", f"${est_price:,.0f}", help=f"Includes {int(premium*100)}% New Build Premium")
    col2.metric("Est. Monthly Rent", f"${est_rent:,.0f}", help="Derived from Rent/SF of 1-Bed comps")
    col3.metric("Sales Comps", f"{len(s_comps)}", f"Avg ${s_comps['Price'].median():,.0f}")
    col4.metric("Rental Comps", f"{len(r_comps)}", f"{radius} mile radius")

    st.divider()

    # 2. MAP & CHARTS SPLIT
    c_left, c_right = st.columns([1.5, 1])

    with c_left:
        st.markdown("#### 🗺️ Market Radius")
        # PYDECK MAP
        view_state = pdk.ViewState(latitude=loc.latitude, longitude=loc.longitude, zoom=12.5)
        
        # Layers
        site_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": loc.latitude, "lon": loc.longitude}]),
            get_position="[lon, lat]", get_color=[255, 0, 0], get_radius=100,
        )
        # The Radius Circle (Polygon)
        # We draw a Scatterplot with a massive radius to simulate the circle
        radius_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": loc.latitude, "lon": loc.longitude}]),
            get_position="[lon, lat]", 
            get_color=[255, 0, 0, 30], # Transparent Red
            get_radius=radius * 1609.34, # Miles to Meters
            stroked=True, get_line_color=[255, 0, 0], line_width_min_pixels=2
        )
        comps_layer = pdk.Layer(
            "ScatterplotLayer",
            data=s_comps,
            get_position="[Lon, Lat]", get_color=[0, 100, 200, 180], get_radius=30,
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            layers=[radius_layer, site_layer, comps_layer], 
            initial_view_state=view_state,
            tooltip={"text": "{Address}\nPrice: ${Price}\nSize: {SqFt} sf"},
            map_style="mapbox://styles/mapbox/light-v9"
        ))

    with c_right:
        st.markdown("#### 📈 Valuation Curve")
        if not s_comps.empty:
            # Prepare clean chart data (Drop NaNs strictly for Altair)
            chart_data = s_comps[['SqFt', 'Price', 'Address', 'YearBuilt']].dropna()
            
            base = alt.Chart(chart_data).mark_circle(size=80, color='#3b82f6', opacity=0.6).encode(
                x=alt.X('SqFt', scale=alt.Scale(zero=False), title='Unit Size (SF)'),
                y=alt.Y('Price', scale=alt.Scale(zero=False), format='$.2s', title='Sold Price'),
                tooltip=['Address', 'Price', 'SqFt', 'YearBuilt']
            )
            
            line = base.transform_regression('SqFt', 'Price').mark_line(color='#9ca3af', strokeDash=[4,4])
            
            target_pt = pd.DataFrame([{'SqFt': target_sf, 'Price': est_price}])
            star = alt.Chart(target_pt).mark_point(shape='star', size=300, color='#ef4444', filled=True).encode(
                x='SqFt', y='Price', tooltip=alt.value("Your Project")
            )
            
            st.altair_chart(base + line + star, use_container_width=True)

    # 3. DETAILED DATA TABLE
    st.markdown("#### 📋 Comparable Sales Detail")
    if not s_comps.empty:
        # Create a professional display table
        display_df = s_comps[['Address', 'Price', 'SqFt', 'YearBuilt', 'dist']].copy()
        display_df['Price/SF'] = display_df['Price'] / display_df['SqFt']
        display_df = display_df.sort_values('dist')
        
        st.dataframe(
            display_df,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%d"),
                "Price/SF": st.column_config.NumberColumn(format="$%.2f"),
                "dist": st.column_config.NumberColumn(label="Dist (mi)", format="%.2f"),
                "YearBuilt": st.column_config.NumberColumn(format="%d"),
            },
            hide_index=True,
            use_container_width=True
        )

else:
    st.info("Enter an address to begin analysis.")
