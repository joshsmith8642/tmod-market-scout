import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import glob
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- 1. CONFIG & STYLE ---
st.set_page_config(layout="wide", page_title="Tmod Intelligence")

st.markdown("""
    <style>
        .reportview-container { margin-top: -2em; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏗️ Tmod Market Intelligence")

# --- 2. PROJECT MANAGER (SAVED SITES) ---
PROJECTS_FILE = "tmod_projects.csv"

def load_projects():
    if os.path.exists(PROJECTS_FILE):
        return pd.read_csv(PROJECTS_FILE)
    return pd.DataFrame(columns=["Name", "Address", "Target_SF"])

def save_project(name, addr, sf):
    df = load_projects()
    new_row = pd.DataFrame([{"Name": name, "Address": addr, "Target_SF sf": sf}])
    # Remove existing if overwriting
    df = df[df["Name"] != name]
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(PROJECTS_FILE, index=False)
    return df

# --- 3. DATA ENGINE ---
@st.cache_data
def load_data():
    redfin_files = [f for f in glob.glob("*.csv") if "Costar" not in f and "requirements" not in f and "tmod_projects" not in f]
    costar_files = glob.glob("*Costar*.csv")
    
    if not redfin_files:
        return None, None

    # Process Sales
    try:
        sales_df = pd.concat([pd.read_csv(f) for f in redfin_files], ignore_index=True)
        sales_df.columns = [c.upper() for c in sales_df.columns]
        
        sales_df = sales_df.dropna(subset=['PRICE', 'LATITUDE'])
        # Strict Numeric Conversion
        cols_to_num = ['PRICE', 'SQUARE FEET', 'LATITUDE', 'LONGITUDE', 'YEAR BUILT']
        for c in cols_to_num:
            if c in sales_df.columns:
                sales_df[c] = pd.to_numeric(sales_df[c], errors='coerce')

        sales_df = sales_df.rename(columns={
            'PRICE': 'Price', 'SQUARE FEET': 'SqFt', 'LATITUDE': 'Lat', 'LONGITUDE': 'Lon', 'YEAR BUILT': 'YearBuilt'
        })
        
        sales_df['PPSF'] = sales_df['Price'] / sales_df['SqFt']
        sales_df = sales_df.dropna(subset=['Price', 'SqFt', 'Lat', 'Lon'])
        # Fill YearBuilt NaNs with 0 to prevent Altair schema errors
        sales_df['YearBuilt'] = sales_df['YearBuilt'].fillna(0).astype(int)
        
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
    st.info("👋 Please upload data files to GitHub to start.")
    st.stop()

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🗂️ Project Manager")
    
    # Load Saved Projects
    projects_df = load_projects()
    project_names = ["New Project"] + projects_df["Name"].tolist()
    
    selected_project = st.selectbox("Select Project", project_names)
    
    # Defaults
    default_addr = "Boise, ID"
    default_sf = 850
    
    if selected_project != "New Project":
        row = projects_df[projects_df["Name"] == selected_project].iloc[0]
        default_addr = row["Address"]
        # Handle case where column might be missing in old CSVs
        if "Target_SF" in row:
             default_sf = int(row["Target_SF"])
        else:
             default_sf = 850

    st.divider()
    
    # Inputs
    project_name_input = st.text_input("Project Name", value=selected_project if selected_project != "New Project" else "")
    address = st.text_input("Target Address", value=default_addr)
    target_sf = st.number_input("Assumption Unit SF", value=default_sf, step=50)
    
    if st.button("💾 Save Project"):
        if project_name_input:
            save_project(project_name_input, address, target_sf)
            st.success(f"Saved {project_name_input}!")
            st.rerun()

    st.divider()
    st.subheader("⚙️ Analysis Settings")
    radius = st.slider("Radius (Miles)", 1, 10, 3)
    premium = st.slider("New Build Premium", 0, 30, 15, format="%d%%") / 100

# --- 5. GEOCODING ---
geolocator = Nominatim(user_agent="tmod_pro_v4")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

try:
    location = geocode(address)
except:
    st.error("⚠️ Geocoding busy. Try again.")
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
    
    # VALUATION LOGIC
    if not s_comps.empty:
        m, b = np.polyfit(s_comps['SqFt'], s_comps['Price'], 1)
        suggested_sale = ((m * target_sf) + b) * (1 + premium)
        
        suggested_rent = 0
        r_count = 0
        if rent_master is not None:
            r_comps = get_comps(rent_master, location.latitude, location.longitude, radius)
            if not r_comps.empty and 'One Bedroom Asking Rent/Unit' in r_comps.columns:
                 base_rent = r_comps['One Bedroom Asking Rent/Unit'].mean()
                 if 'One Bedroom Avg SF' in r_comps.columns:
                     avg_sf = r_comps['One Bedroom Avg SF'].mean()
                     if pd.notnull(avg_sf) and avg_sf > 0:
                        base_rent = base_rent * (target_sf / avg_sf)
                 suggested_rent = base_rent * (1 + premium)
                 r_count = len(r_comps)

        # --- DASHBOARD UI ---
        
        # 1. HERO METRICS
        st.subheader(f"Valuation: {project_name_input if project_name_input else address}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. Sale Price", f"${suggested_sale:,.0f}")
        m2.metric("Est. PPSF", f"${suggested_sale/target_sf:.0f}/sf")
        m3.metric("Est. Monthly Rent", f"${suggested_rent:,.0f}" if suggested_rent > 0 else "N/A")
        m4.metric("Active Comps", f"{len(s_comps)} Sales / {r_count} Rentals")

        # 2. VISUAL MAP (PYDECK)
        st.markdown("### 🗺️ Radius Map")
        
        # Calculate Radius in Meters for the map
        radius_meters = radius * 1609.34
        
        # Layers
        target_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": location.latitude, "lon": location.longitude}]),
            get_position="[lon, lat]",
            get_color=[255, 0, 0, 200], # Red
            get_radius=50,
            pickable=True,
        )
        
        radius_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": location.latitude, "lon": location.longitude}]),
            get_position="[lon, lat]",
            get_color=[255, 0, 0, 30], # Translucent Red
            get_radius=radius_meters,
            stroked=True,
            get_line_color=[255, 0, 0, 100],
            line_width_min_pixels=2,
        )
        
        comps_layer = pdk.Layer(
            "ScatterplotLayer",
            data=s_comps,
            get_position="[Lon, Lat]",
            get_color=[0, 128, 255, 150], # Blue dots
            get_radius=30,
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=location.latitude,
            longitude=location.longitude,
            zoom=12 if radius > 5 else 13,
            pitch=0,
        )

        r = pdk.Deck(
            layers=[radius_layer, target_layer, comps_layer],
            initial_view_state=view_state,
            tooltip={"text": "{ADDRESS}\n${Price}"},
            map_style="mapbox://styles/mapbox/light-v9"
        )
        st.pydeck_chart(r)

        # 3. CHARTS & DATA
        st.divider()
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("#### 📈 Market Regression")
            
            # Prepare clean data for Altair to prevent Schema Error
            chart_data = s_comps.copy()
            # Ensure no NaNs in plotting columns
            chart_data = chart_data.dropna(subset=['SqFt', 'Price', 'YearBuilt'])
            
            base = alt.Chart(chart_data).mark_circle(size=60, opacity=0.6, color='#3182bd').encode(
                x=alt.X('SqFt', scale=alt.Scale(zero=False), title='Square Feet'),
                y=alt.Y('Price', scale=alt.Scale(zero=False), title='Sold Price', format='$.2s'),
                tooltip=[
                    alt.Tooltip('ADDRESS', title='Address'),
                    alt.Tooltip('Price', format='$,.0f'),
                    alt.Tooltip('SqFt', format=',.0f'),
                    alt.Tooltip('YearBuilt', title='Year Built')
                ]
            )
            
            line = base.transform_regression('SqFt', 'Price').mark_line(color='gray', strokeDash=[5,5])
            
            target_pt = pd.DataFrame([{'SqFt': target_sf, 'Price': suggested_sale}])
            star = alt.Chart(target_pt).mark_point(shape='star', size=400, color='#e6550d', filled=True).encode(
                x='SqFt', y='Price', tooltip=alt.value("Your Project")
            )
            
            st.altair_chart(base + line + star, use_container_width=True)

        with c2:
            st.markdown("#### 📋 Closest Comps")
            st.dataframe(
                s_comps[['ADDRESS', 'Price', 'SqFt', 'dist', 'YearBuilt']]
                .sort_values('dist')
                .head(5)
                .style.format({'Price': '${:,.0f}', 'dist': '{:.2f} mi', 'YearBuilt': '{:.0f}'}),
                hide_index=True
            )

    else:
        st.warning(f"No sales data found within {radius} miles.")
