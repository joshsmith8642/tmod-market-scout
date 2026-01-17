import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import glob
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- 1. CONFIG & STYLE ---
st.set_page_config(layout="wide", page_title="Tmod Intelligence")

# Custom CSS to hide default "Made with Streamlit" footer and tighten spacing
st.markdown("""
    <style>
        .reportview-container { margin-top: -2em; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title(" Tmod Market Intelligence")

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    # Load Redfin (Sales)
    redfin_files = [f for f in glob.glob("*.csv") if "Costar" not in f and "requirements" not in f]
    costar_files = glob.glob("*Costar*.csv")
    
    if not redfin_files:
        return None, None

    # Process Sales
    try:
        sales_df = pd.concat([pd.read_csv(f) for f in redfin_files], ignore_index=True)
        # FORCE UPPERCASE COLUMNS to avoid mismatch errors
        sales_df.columns = [c.upper() for c in sales_df.columns]
        
        # Clean & Convert
        sales_df = sales_df.dropna(subset=['PRICE', 'LATITUDE'])
        sales_df['Price'] = pd.to_numeric(sales_df['PRICE'], errors='coerce')
        sales_df['SqFt'] = pd.to_numeric(sales_df['SQUARE FEET'], errors='coerce')
        sales_df['Lat'] = pd.to_numeric(sales_df['LATITUDE'], errors='coerce')
        sales_df['Lon'] = pd.to_numeric(sales_df['LONGITUDE'], errors='coerce')
        
        # Safe handling for Year Built (fill NaNs with 0 to prevent crashes)
        if 'YEAR BUILT' in sales_df.columns:
            sales_df['YEAR BUILT'] = sales_df['YEAR BUILT'].fillna(0).astype(int)
        
        sales_df = sales_df.dropna(subset=['Price', 'SqFt', 'Lat', 'Lon'])
    except:
        return None, None

    # Process Rent
    rent_df = None
    if costar_files:
        try:
            rent_df = pd.read_csv(costar_files[0])
            rent_df = rent_df.rename(columns={'Latitude': 'Lat', 'Longitude': 'Lon'})
        except:
            pass

    return sales_df, rent_df

sales_master, rent_master = load_data()

if sales_master is None:
    st.info("👋 Welcome! Please upload your CSV data files to the GitHub repository to begin.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("📍 Project Specs")
    address = st.text_input("Target Address", "Boise, ID")
    target_sf = st.number_input("Assumption Unit SF", value=850, step=50)
    
    st.divider()
    st.subheader("⚙️ Market Settings")
    radius = st.slider("Search Radius (Miles)", 1, 10, 3)
    premium = st.slider("New Build Premium", 0, 30, 15, format="%d%%") / 100

# --- 4. GEOCODING & LOGIC ---
geolocator = Nominatim(user_agent="tmod_pro_v1")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

try:
    location = geocode(address)
except:
    st.error("⚠️ Geocoding service busy. Please wait 5 seconds and try again.")
    st.stop()

if location:
    # SPATIAL FILTER
    def get_comps(df, t_lat, t_lon, r_miles):
        R = 3958.8 
        dlat, dlon = np.radians(df['Lat'] - t_lat), np.radians(df['Lon'] - t_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(t_lat)) * np.cos(np.radians(df['Lat'])) * np.sin(dlon/2)**2
        df['dist'] = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))
        return df[df['dist'] <= r_miles].copy()

    s_comps = get_comps(sales_master, location.latitude, location.longitude, radius)
    
    # CALCULATE VALUATIONS
    if not s_comps.empty:
        # Sales Math
        m, b = np.polyfit(s_comps['SqFt'], s_comps['Price'], 1)
        suggested_sale = ((m * target_sf) + b) * (1 + premium)
        
        # Rent Math
        suggested_rent = 0
        r_count = 0
        if rent_master is not None:
            r_comps = get_comps(rent_master, location.latitude, location.longitude, radius)
            if not r_comps.empty and 'One Bedroom Asking Rent/Unit' in r_comps.columns:
                 # Simple avg of 1-Beds for robustness
                 base_rent = r_comps['One Bedroom Asking Rent/Unit'].mean()
                 # Adjust for size diff (simple ratio) if SqFt available
                 if 'One Bedroom Avg SF' in r_comps.columns:
                     avg_sf = r_comps['One Bedroom Avg SF'].mean()
                     base_rent = base_rent * (target_sf / avg_sf)
                 
                 suggested_rent = base_rent * (1 + premium)
                 r_count = len(r_comps)

        # --- 5. DASHBOARD UI ---
        
        # SECTION A: HERO METRICS
        st.subheader(f"Valuation: {location.address}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. Sale Price", f"${suggested_sale:,.0f}", help="Includes new construction premium")
        m2.metric("Est. PPSF", f"${suggested_sale/target_sf:.0f}/sf")
        if suggested_rent > 0:
            m3.metric("Est. Monthly Rent", f"${suggested_rent:,.0f}")
        else:
            m3.metric("Est. Monthly Rent", "N/A")
        m4.metric("Comps Found", f"{len(s_comps)} Sales / {r_count} Rentals")
        
        st.divider()

        # SECTION B: CHARTS
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("#### 📈 Price vs. Size Regression")
            # Base Chart (The Comps)
            # FIXED: Updated tooltip to use 'YEAR BUILT' (uppercase) to match data
            base = alt.Chart(s_comps).mark_circle(size=60, opacity=0.6, color='#3182bd').encode(
                x=alt.X('SqFt', scale=alt.Scale(zero=False), title='Square Feet'),
                y=alt.Y('Price', scale=alt.Scale(zero=False), title='Sold Price', format='$.2s'),
                tooltip=['ADDRESS', 'Price', 'SqFt', 'YEAR BUILT']
            )
            
            # The Regression Line
            line = base.transform_regression('SqFt', 'Price').mark_line(color='gray', strokeDash=[5,5])
            
            # The "YOU ARE HERE" Star
            target_data = pd.DataFrame({'SqFt': [target_sf], 'Price': [suggested_sale]})
            target = alt.Chart(target_data).mark_point(shape='star', size=400, color='#e6550d', filled=True).encode(
                x='SqFt', y='Price', tooltip=alt.value("Your Project")
            )
            
            st.altair_chart(base + line + target, use_container_width=True)

        with c2:
            st.markdown("#### 📋 Top 5 Comparable Sales")
            # Clean Table
            display_cols = ['ADDRESS', 'Price', 'SqFt', 'dist']
            st.dataframe(
                s_comps[display_cols].sort_values('dist').head(5).style.format({'Price': '${:,.0f}', 'dist': '{:.1f} mi'}),
                hide_index=True
            )

    else:
        st.warning(f"No comps found within {radius} miles. Try increasing the radius.")

else:
    st.info("Enter an address to generate valuation.")
