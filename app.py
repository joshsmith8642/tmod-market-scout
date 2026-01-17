import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import glob
from geopy.geocoders import Nominatim

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Tmod Intelligence")

# --- DEBUGGING HEADER ---
st.title("Tmod Market Intelligence")

@st.cache_data
def load_data():
    # 1. Look for Redfin CSVs in the MAIN folder (root)
    # We exclude the costar file and any requirements file
    files = [f for f in glob.glob("*.csv") if "Costar" not in f and "requirements" not in f]
    
    if not files:
        return None, None
    
    st.toast(f"Found {len(files)} Redfin files", icon="✅")
    
    # Load and combine them
    sales_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Clean up columns
    sales_df = sales_df.dropna(subset=['PRICE', 'LATITUDE'])
    sales_df = sales_df.rename(columns={'PRICE': 'Price', 'SQUARE FEET': 'SqFt', 'LATITUDE': 'Lat', 'LONGITUDE': 'Lon'})
    
    # 2. Look for CoStar CSV in the MAIN folder
    costar_files = glob.glob("*Costar*.csv")
    if not costar_files:
        st.error("Could not find the CoStar file. Make sure it has 'Costar' in the name.")
        return sales_df, None
        
    rent_df = pd.read_csv(costar_files[0]).rename(columns={'Latitude': 'Lat', 'Longitude': 'Lon'})
    
    return sales_df, rent_df

# --- LOAD ---
try:
    sales_master, rent_master = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

if sales_master is None:
    st.warning("⚠️ No data found. Please drag your CSV files into the GitHub repository main page.")
    st.stop()

# --- APP LOGIC ---
def get_radius_comps(df, t_lat, t_lon, r_miles):
    R = 3958.8 
    dlat, dlon = np.radians(df['Lat'] - t_lat), np.radians(df['Lon'] - t_lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(t_lat)) * np.cos(np.radians(df['Lat'])) * np.sin(dlon/2)**2
    df['dist'] = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))
    return df[df['dist'] <= r_miles].copy()

# Sidebar
with st.sidebar:
    st.header("Project Inputs")
    address = st.text_input("Target Address", "Boise, ID")
    target_sf = st.number_input("Assumption SF", value=850)
    radius = st.slider("Radius (mi)", 1, 10, 3)
    premium = st.slider("New Construction Premium", 1.0, 1.3, 1.15)

# Main Dashboard
geolocator = Nominatim(user_agent="tmod_web_app")
location = geolocator.geocode(address)

if location:
    st.subheader(f"Analysis for: {location.address}")
    
    s_comps = get_radius_comps(sales_master, location.latitude, location.longitude, radius)
    
    if s_comps.empty:
        st.warning("No sales comps found in this radius.")
    else:
        # Sales Valuation
        m, b = np.polyfit(s_comps['SqFt'], s_comps['Price'], 1)
        suggested_price = ((m * target_sf) + b) * premium
        
        c1, c2 = st.columns(2)
        c1.metric("Suggested Sale Price", f"${suggested_price:,.0f}", f"${suggested_price/target_sf:.2f}/sf")
        
        # Chart
        chart = alt.Chart(s_comps).mark_circle(size=60).encode(
            x='SqFt', y='Price', tooltip=['Price', 'SqFt', 'ADDRESS']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    # Rental Section (Only if CoStar exists)
    if rent_master is not None:
        r_comps = get_radius_comps(rent_master, location.latitude, location.longitude, radius)
        if not r_comps.empty:
             # Basic Rental Avg for now (robust logic relies on 1-bed columns existing)
             avg_rent = r_comps['One Bedroom Asking Rent/Unit'].mean()
             if pd.notnull(avg_rent):
                 st.metric("Avg 1-Bed Market Rent", f"${avg_rent:,.0f}")
