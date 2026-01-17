import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import glob
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- 1. CONFIG ---
st.set_page_config(layout="wide", page_title="Tmod Intelligence")
st.title("Tmod Market Intelligence")

# --- 2. ROBUST DATA LOADER ---
# We keep the cache for speed, but removed the visual 'st.toast' that caused the crash
@st.cache_data
def load_data():
    # LOOK FOR FILES IN ROOT FOLDER
    redfin_files = [f for f in glob.glob("*.csv") if "Costar" not in f and "requirements" not in f]
    costar_files = glob.glob("*Costar*.csv")
    
    if not redfin_files:
        return None, None

    # LOAD REDFIN (SALES)
    try:
        sales_df = pd.concat([pd.read_csv(f) for f in redfin_files], ignore_index=True)
        
        # Standardize Columns
        sales_df.columns = [c.upper() for c in sales_df.columns]
        
        # Clean Junk Rows
        sales_df = sales_df.dropna(subset=['PRICE', 'LATITUDE'])
        
        # Create Analysis Columns
        sales_df['Price'] = pd.to_numeric(sales_df['PRICE'], errors='coerce')
        sales_df['SqFt'] = pd.to_numeric(sales_df['SQUARE FEET'], errors='coerce')
        sales_df['Lat'] = pd.to_numeric(sales_df['LATITUDE'], errors='coerce')
        sales_df['Lon'] = pd.to_numeric(sales_df['LONGITUDE'], errors='coerce')
        
        sales_df = sales_df.dropna(subset=['Price', 'SqFt', 'Lat', 'Lon'])
        
    except Exception:
        return None, None

    # LOAD COSTAR (RENT)
    rent_df = None
    if costar_files:
        try:
            rent_df = pd.read_csv(costar_files[0])
            rent_df = rent_df.rename(columns={'Latitude': 'Lat', 'Longitude': 'Lon'})
        except Exception:
            pass

    return sales_df, rent_df

# --- 3. LOAD DATA ---
sales_master, rent_master = load_data()

if sales_master is None:
    st.warning("⚠️ No data found. Please upload your CSV files to the GitHub repository.")
    st.stop()

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Project Inputs")
    address = st.text_input("Target Address", "Boise, ID")
    target_sf = st.number_input("Assumption SF", value=850)
    radius = st.slider("Radius (mi)", 1, 10, 3)
    premium = st.slider("New Construction Premium", 1.0, 1.3, 1.15)

# --- 5. GEOCODING ENGINE ---
geolocator = Nominatim(user_agent="tmod_app_v3")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

location = None
try:
    location = geocode(address)
except Exception as e:
    st.error(f"Geocoding Error: {e}")

# --- 6. DASHBOARD LOGIC ---
if location:
    st.subheader(f"Analysis for: {location.address}")
    
    # FILTER FUNCTION
    def get_radius_comps(df, t_lat, t_lon, r_miles):
        R = 3958.8 
        dlat, dlon = np.radians(df['Lat'] - t_lat), np.radians(df['Lon'] - t_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(t_lat)) * np.cos(np.radians(df['Lat'])) * np.sin(dlon/2)**2
        df['dist'] = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))
        return df[df['dist'] <= r_miles].copy()

    s_comps = get_radius_comps(sales_master, location.latitude, location.longitude, radius)
    
    # METRICS
    if not s_comps.empty:
        # Sales Valuation
        m, b = np.polyfit(s_comps['SqFt'], s_comps['Price'], 1)
        suggested_price = ((m * target_sf) + b) * premium
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Suggested Sale Price", f"${suggested_price:,.0f}", f"${suggested_price/target_sf:.2f}/sf")
        c1.caption(f"Based on {len(s_comps)} sales comps")
        
        # Rental Valuation
        if rent_master is not None:
            r_comps = get_radius_comps(rent_master, location.latitude, location.longitude, radius)
            if not r_comps.empty:
                # Robust Rent Logic
                if 'One Bedroom Asking Rent/Unit' in r_comps.columns:
                     avg_rent = r_comps['One Bedroom Asking Rent/Unit'].mean()
                     c2.metric("Avg 1-Bed Rent", f"${avg_rent:,.0f}")
                     c2.caption(f"Based on {len(r_comps)} rental comps")
                else:
                    c2.info("Rent data found but missing 'One Bedroom' column.")
            else:
                c2.warning("No rental comps in radius")
        
        # Charts
        tab1, tab2 = st.tabs(["Regression", "Raw Data"])
        with tab1:
            chart = alt.Chart(s_comps).mark_circle(size=60).encode(
                x=alt.X('SqFt', scale=alt.Scale(zero=False)),
                y=alt.Y('Price', scale=alt.Scale(zero=False)),
                tooltip=['ADDRESS', 'Price', 'SqFt']
            ).interactive()
            
            # Add the 'Target' Star
            target_df = pd.DataFrame({'SqFt': [target_sf], 'Price': [suggested_price]})
            star = alt.Chart(target_df).mark_point(shape='star', size=300, color='red', filled=True).encode(
                x='SqFt', y='Price', tooltip=alt.value("Your Project")
            )
            
            st.altair_chart(chart + star, use_container_width=True)
            
        with tab2:
            st.dataframe(s_comps[['ADDRESS', 'Price', 'SqFt', 'dist']].sort_values('dist'))

    else:
        st.error(f"No sales data found within {radius} miles of this location.")
else:
    st.info("Enter a valid address in the sidebar to begin.")
