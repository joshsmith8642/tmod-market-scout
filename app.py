import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import glob
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- 1. CONFIG & STYLING ---
st.set_page_config(layout="wide", page_title="Tmod Market Scout")

# Professional Styling: Removes whitespace, tightens metrics
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        .stMetric { background-color: #F9FAFB; border: 1px solid #E5E7EB; padding: 10px; border-radius: 8px; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; color: #111827; font-weight: 600; }
        div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #6B7280; }
        h1, h2, h3 { color: #1F2937; letter-spacing: -0.025em; }
    </style>
""", unsafe_allow_html=True)

# --- 2. PROJECT MANAGER (SAVED SITES) ---
PROJECTS_FILE = "tmod_projects.csv"

def load_projects():
    if os.path.exists(PROJECTS_FILE):
        try:
            return pd.read_csv(PROJECTS_FILE)
        except:
            pass
    return pd.DataFrame(columns=["Name", "Address", "Target_SF"])

def save_project(name, addr, sf):
    df = load_projects()
    # Remove existing entry if updating the same name
    df = df[df["Name"] != name]
    new_row = pd.DataFrame([{"Name": name, "Address": addr, "Target_SF": sf}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(PROJECTS_FILE, index=False)
    return df

# --- 3. DATA ENGINE ---
@st.cache_data
def load_data():
    # Load Redfin (Sales)
    # Checks root directory for any CSV that looks like Redfin data
    redfin_files = [f for f in glob.glob("*.csv") if "Costar" not in f and "requirements" not in f and "tmod" not in f]
    costar_files = glob.glob("*Costar*.csv")
    
    sales_df = None
    rent_df = None

    # SALES PROCESSING
    if redfin_files:
        try:
            # Load & Concat
            df = pd.concat([pd.read_csv(f) for f in redfin_files], ignore_index=True)
            df.columns = [c.upper() for c in df.columns] # Force Uppercase
            
            # Map columns
            df = df.rename(columns={
                'PRICE': 'Price', 'SQUARE FEET': 'SqFt', 
                'LATITUDE': 'Lat', 'LONGITUDE': 'Lon', 'YEAR BUILT': 'YearBuilt',
                'ADDRESS': 'Address'
            })
            
            # Coerce Numerics
            for c in ['Price', 'SqFt', 'Lat', 'Lon', 'YearBuilt']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # Strict Cleaning
            df = df.dropna(subset=['Price', 'SqFt', 'Lat', 'Lon'])
            
            # Safe Year handling
            if 'YearBuilt' in df.columns:
                df['YearBuilt'] = df['YearBuilt'].fillna(0).astype(int)
            else:
                df['YearBuilt'] = 0
                
            sales_df = df
        except Exception:
            pass # Fail silently, return None

    # RENT PROCESSING
    if costar_files:
        try:
            df = pd.read_csv(costar_files[0])
            df = df.rename(columns={'Latitude': 'Lat', 'Longitude': 'Lon'})
            rent_df = df
        except Exception:
            pass

    return sales_df, rent_df

sales_master, rent_master = load_data()

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("Tmod Scout 🏢")
    
    # --- Saved Sites Logic ---
    projects_df = load_projects()
    project_list = ["Create New..."] + projects_df["Name"].tolist()
    
    st.subheader("📁 Project")
    selected_project = st.selectbox("Load Saved Site", project_list, label_visibility="collapsed")
    
    # Input Variables
    p_name, p_addr, p_sf = "", "Boise, ID", 850
    
    if selected_project != "Create New...":
        row = projects_df[projects_df["Name"] == selected_project].iloc[0]
        p_name = row["Name"]
        p_addr = row["Address"]
        p_sf = int(row["Target_SF"])

    name_input = st.text_input("Project Name", value=p_name, placeholder="e.g. Tmod Downtown")
    address = st.text_input("Address", value=p_addr)
    target_sf = st.number_input("Target Unit SF", value=p_sf, step=50)
    
    if st.button("💾 Save Project", use_container_width=True):
        if name_input and address:
            save_project(name_input, address, target_sf)
            st.toast(f"Saved {name_input}!", icon="✅")
            st.rerun()

    st.divider()
    st.subheader("⚙️ Settings")
    radius = st.slider("Radius (Miles)", 0.5, 10.0, 3.0, 0.5)
    premium = st.slider("New Construction Premium", 0, 30, 15, format="%d%%") / 100

# --- 5. MAIN LOGIC ---
if sales_master is None:
    st.info("👋 To begin, upload your Redfin/Costar CSV files to the GitHub repository.")
    st.stop()

# Geocoding
geolocator = Nominatim(user_agent="tmod_pro_v5")
try:
    loc = geolocator.geocode(address)
except:
    st.error("⚠️ Geocoding service is busy. Please verify the address or try again in 5 seconds.")
    st.stop()

if loc:
    # SPATIAL FILTER
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

    # VALUATION MATH
    est_price = 0
    if not s_comps.empty:
        # Linear Regression
        m, b = np.polyfit(s_comps['SqFt'], s_comps['Price'], 1)
        est_price = ((m * target_sf) + b) * (1 + premium)

    est_rent = 0
    if not r_comps.empty and 'One Bedroom Asking Rent/Unit' in r_comps.columns:
        # Rent/SF Logic based on 1-Bed anchor
        valid_rents = r_comps.dropna(subset=['One Bedroom Asking Rent/Unit', 'One Bedroom Avg SF'])
        if not valid_rents.empty:
            avg_rent_sf = (valid_rents['One Bedroom Asking Rent/Unit'] / valid_rents['One Bedroom Avg SF']).median()
            est_rent = (avg_rent_sf * target_sf) * (1 + premium)

    # --- DASHBOARD HEADER ---
    st.subheader(f"{name_input if name_input else address}")
    
    # HERO METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Est. Sale Price", f"${est_price:,.0f}", help="Includes New Construction Premium")
    m2.metric("Est. Monthly Rent", f"${est_rent:,.0f}")
    m3.metric("Sales Comps", f"{len(s_comps)}", f"Avg ${s_comps['Price'].median():,.0f}")
    m4.metric("Rental Comps", f"{len(r_comps)}", f"Radius: {radius} mi")

    st.divider()

    # --- SPLIT LAYOUT: MAP & CHART ---
    c_left, c_right = st.columns([1.5, 1])

    with c_left:
        st.markdown("#### 🗺️ Market Radius")
        # --- PYDECK MAP ENGINE ---
        # 1. Radius Circle (Polygon)
        radius_meters = radius * 1609.34
        
        # 2. Map Layers
        layers = []
        
        # Site Location (Red Dot)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": loc.latitude, "lon": loc.longitude}]),
            get_position="[lon, lat]", get_color=[239, 68, 68, 255], get_radius=120,
        ))
        
        # Radius Ring (Translucent Red)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": loc.latitude, "lon": loc.longitude}]),
            get_position="[lon, lat]", 
            get_color=[239, 68, 68, 20], # Very transparent
            get_radius=radius_meters,
            stroked=True, get_line_color=[239, 68, 68, 150], line_width_min_pixels=2
        ))
        
        # Comps (Blue Dots)
        if not s_comps.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=s_comps,
                get_position="[Lon, Lat]", get_color=[59, 130, 246, 160], get_radius=40,
                pickable=True, auto_highlight=True
            ))

        # Map View
        view_state = pdk.ViewState(latitude=loc.latitude, longitude=loc.longitude, zoom=12.5)
        
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{Address}\nPrice: ${Price}\nBuilt: {YearBuilt}"},
            map_style="mapbox://styles/mapbox/light-v9"
        ))

    with c_right:
        st.markdown("#### 📈 Price Regression")
        
        if not s_comps.empty:
            # --- ALTAIR CHART SANITIZER ---
            # This block prevents the SchemaValidationError
            # 1. Select only needed columns
            chart_data = s_comps[['SqFt', 'Price', 'Address', 'YearBuilt']].copy()
            # 2. Force types to pure Python types (No numpy ints)
            chart_data['YearBuilt'] = chart_data['YearBuilt'].astype(str) 
            chart_data = chart_data.dropna()
            
            # Base Chart
            base = alt.Chart(chart_data).mark_circle(size=80, color='#3b82f6', opacity=0.6).encode(
                x=alt.X('SqFt', scale=alt.Scale(zero=False), title='Unit Size (SF)'),
                y=alt.Y('Price', scale=alt.Scale(zero=False), format='$.2s', title='Sold Price'),
                tooltip=['Address', 'Price', 'SqFt', 'YearBuilt']
            )
            
            # Regression Line
            line = base.transform_regression('SqFt', 'Price').mark_line(color='#9ca3af', strokeDash=[4,4])
            
            # Target Star
            target_pt = pd.DataFrame([{'SqFt': float(target_sf), 'Price': float(est_price)}])
            star = alt.Chart(target_pt).mark_point(shape='star', size=350, color='#ef4444', filled=True).encode(
                x='SqFt', y='Price', tooltip=alt.value("Your Project")
            )
            
            st.altair_chart(base + line + star, use_container_width=True)
        else:
            st.warning("No comps available for regression.")

    # --- DATA TABLE ---
    st.markdown("#### 📋 Comparable Sales List")
    if not s_comps.empty:
        # Format for display
        display_df = s_comps[['Address', 'Price', 'SqFt', 'YearBuilt', 'dist']].copy()
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.0f}")
        display_df['SqFt'] = display_df['SqFt'].apply(lambda x: f"{x:,.0f}")
        display_df['YearBuilt'] = display_df['YearBuilt'].astype(str).replace('0', 'N/A')
        
        st.dataframe(
            display_df.sort_values('dist'),
            column_config={
                "dist": st.column_config.NumberColumn("Distance (mi)", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )

else:
    st.info("Enter an address to begin.")
