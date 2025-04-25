# -----------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import folium
import json
import geopandas as gpd
from datetime import datetime
from math import radians
from shapely.geometry import MultiPoint, Point
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from pydeck.types import String
import matplotlib.pyplot as plt
import openai
from openai import OpenAI
from matplotlib.patches import Patch
from io import BytesIO
from geopy.distance import geodesic
import joblib
import contextily as ctx
import requests
# -----------------------------------------------------------------------------
# Sidebar Page Selection
# -----------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>AI City Scout</h1>", unsafe_allow_html=True)
page = st.sidebar.selectbox("Select Page", ["Future Predictions", "Social Media Detector", "Past Disasters"])

# -----------------------------------------------------------------------------
# Load and Cache Datasets
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    
    disaster_events = pd.read_csv("https://raw.githubusercontent.com/YuwenAprilYang/Projects/1750b61661e3bf15e72b102a42c20f65a1805171/AI%20City%20Scout/disaster_events.csv")
    sensor_df = pd.read_csv("https://raw.githubusercontent.com/YuwenAprilYang/Projects/1750b61661e3bf15e72b102a42c20f65a1805171/AI%20City%20Scout/sensor_readings.csv")
    weather_df = pd.read_csv("https://raw.githubusercontent.com/YuwenAprilYang/Projects/1750b61661e3bf15e72b102a42c20f65a1805171/AI%20City%20Scout/weather_historical.csv")
    geo_url = "https://raw.githubusercontent.com/YuwenAprilYang/Projects/1750b61661e3bf15e72b102a42c20f65a1805171/AI%20City%20Scout/city_map.geojson"
    response = requests.get(geo_url)

    # Check for failure
    if response.status_code != 200:
        raise FileNotFoundError(f"Failed to fetch GeoJSON: {geo_url}")

    # Convert to GeoDataFrame
    infra_geojson = response.json()
    infra_gdf = gpd.GeoDataFrame.from_features(infra_geojson['features'])
    infra_gdf["lon"] = infra_gdf.geometry.x
    infra_gdf["lat"] = infra_gdf.geometry.y
    return disaster_events, sensor_df, weather_df, infra_gdf

disaster_events, sensor_df, weather_df, infra_gdf = load_data()

# Convert the date column to datetime type and extract year and month for filtering
disaster_events['date'] = pd.to_datetime(disaster_events['date'])
disaster_events['year'] = disaster_events['date'].dt.year
disaster_events['month'] = disaster_events['date'].dt.month

# -----------------------------------------------------------------------------
# Process Sensors and Detect Outliers
# -----------------------------------------------------------------------------
@st.cache_data
def process_sensor_data(_sensor_df, _weather_df):
    sensor_df = _sensor_df.copy()
    weather_df = _weather_df.copy()

    # Convert timestamp strings to datetime objects
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])

    # Create additional columns for date and hour extraction
    sensor_df['date'] = sensor_df['timestamp'].dt.date
    sensor_df['hour'] = sensor_df['timestamp'].dt.hour
    weather_df['date'] = weather_df['timestamp'].dt.date
    weather_df['hour'] = weather_df['timestamp'].dt.hour

    # Specify the sensor types to process
    sensor_types = ["flood", "humidity", "seismic", "temp"] 
    clean_sensors = pd.DataFrame()
    for sensor_type in sensor_types:
        sensor_subset = sensor_df[sensor_df["sensor_type"] == sensor_type].copy()
        if sensor_subset.empty:
            continue

        # Use the Isolation Forest to detect anomalies (outliers)
        clf = IsolationForest(contamination=0.05, random_state=42)
        readings_normalized = (sensor_subset["reading_value"] - sensor_subset["reading_value"].mean()) / sensor_subset["reading_value"].std()
        sensor_subset["anomaly"] = clf.fit_predict(readings_normalized.values.reshape(-1, 1))
        
        # Keep only the data predicted as normal (anomaly == 1)
        clean_subset = sensor_subset[sensor_subset["anomaly"] == 1]
        clean_sensors = pd.concat([clean_sensors, clean_subset], ignore_index=True)
    return clean_sensors, sensor_df, weather_df

clean_sensors, sensor_df, weather_df = process_sensor_data(sensor_df, weather_df)

# -----------------------------------------------------------------------------
# Load and Process Weather Data
# -----------------------------------------------------------------------------
@st.cache_data
def process_weather_data(_weather_df):
    weather = _weather_df.copy()

    # Select weather data within a specific period
    current_weather = weather[
        (weather["timestamp"] >= "2023-01-01") & 
        (weather["timestamp"] <= "2023-02-04")
    ]

    # Calculate cumulative precipitation over each 3-hour period
    current_weather["precip_3h"] = current_weather.groupby(
        pd.Grouper(key="timestamp", freq="3H")
    )["precipitation_mm"].transform("sum")
    current_weather["pressure_drop"] = (
        current_weather["pressure_hPa"].rolling(window=6).max() - current_weather["pressure_hPa"]
    )
    current_weather["pressure_drop_flag"] = np.where(current_weather["pressure_drop"] > 10, 1, 0)
    return current_weather

current_weather = process_weather_data(weather_df)

# -----------------------------------------------------------------------------
# Disaster Classification Function
# -----------------------------------------------------------------------------
def classify_disaster(row, weather_row):
    """Classify the disaster based on sensor type and threshold logic."""
    if row['sensor_type'] == 'flood' and (
        row['reading_value'] > 80 or
        (weather_row['precipitation_mm'] > 100 and weather_row['humidity_%'] > 85)
    ):
        return 'Flood'
    elif row['sensor_type'] == 'seismic' and row['reading_value'] > 40:
        return 'Earthquake'
    elif row['sensor_type'] == 'temp' and row['reading_value'] > 45 and weather_row['humidity_%'] < 25:
        return 'Fire'
    return None

# -----------------------------------------------------------------------------
# Generate Interactive Disaster Map (Past Disasters)
# -----------------------------------------------------------------------------
@st.cache_resource
def generate_disaster_map(date: str, hour: int):
    target_date = pd.to_datetime(date).date()
    target_hour = int(hour)

    # Filter sensor and weather data by date and hour
    sensor_sub = sensor_df[(sensor_df['date'] == target_date) & (sensor_df['hour'] == target_hour)].copy()
    weather_sub = weather_df[(weather_df['date'] == target_date) & (weather_df['hour'] == target_hour)].copy()
    
    if weather_sub.empty or sensor_sub.empty:
        st.warning("⚠️ Data for the selected time does not exist.")
        return None
    
    weather_row = weather_sub.iloc[0]
    sensor_sub['disaster_type'] = sensor_sub.apply(lambda row: classify_disaster(row, weather_row), axis=1)
    hurricane = (weather_row['wind_speed_kmph'] > 100) and (weather_row['pressure_hPa'] < 980)

    # Determine default map center from sensor latitude and longitude if available
    if 'latitude' in sensor_sub.columns and 'longitude' in sensor_sub.columns:
        default_lat = sensor_sub['latitude'].mean()
        default_lon = sensor_sub['longitude'].mean()
    else:
        default_lat, default_lon = 0, 0  

    # Create a folium map centered at the default coordinates
    folium_map = folium.Map(location=[default_lat, default_lon], zoom_start=8)
    
    # Add markers for each disaster point detected by sensors
    for idx, row in sensor_sub.iterrows():
        if row['disaster_type'] is not None:
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Disaster: {row['disaster_type']}\nValue: {row['reading_value']}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(folium_map)
   
   # Add a marker for hurricane if detected
    if hurricane:
        folium.Marker(
            location=[default_lat, default_lon],
            popup="Hurricane Detected!",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(folium_map)
    return folium_map

# -----------------------------------------------------------------------------
# Generate Static Disaster Map with Basemap (Future Predictions)
# -----------------------------------------------------------------------------
@st.cache_resource
def generate_static_disaster_map(date: str, hour: int):
    """Generate a static map image with a basemap via contextily."""
    target_date = pd.to_datetime(date).date()
    target_hour = int(hour)
    sensor_sub = sensor_df[(sensor_df['date'] == target_date) & (sensor_df['hour'] == target_hour)].copy()
    weather_sub = weather_df[(weather_df['date'] == target_date) & (weather_df['hour'] == target_hour)].copy()
    if weather_sub.empty or sensor_sub.empty:
        st.warning("⚠️ Data for the selected time does not exist.")
        return None

    weather_row = weather_sub.iloc[0]
    sensor_sub['disaster_type'] = sensor_sub.apply(lambda row: classify_disaster(row, weather_row), axis=1)
    hurricane = (weather_row['wind_speed_kmph'] > 100) and (weather_row['pressure_hPa'] < 980)
    
    # Filter sensor data to include only points where a disaster was detected
    sensor_disasters = sensor_sub[sensor_sub['disaster_type'].notnull()]
    if sensor_disasters.empty:
        st.warning("No disasters detected for the selected time.")
        return None

    # Convert the sensor disaster points to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        sensor_disasters,
        geometry=gpd.points_from_xy(sensor_disasters['longitude'], sensor_disasters['latitude']),
        crs="EPSG:4326"
    )

    # Define the color mapping for different disaster types
    disaster_colors = {
        "Flood": "red",
        "Earthquake": "orange",
        "Fire": "purple"
    }
    
    # Calculate the map center based on the mean of the coordinates
    default_lat = gdf['latitude'].mean()
    default_lon = gdf['longitude'].mean()
    
    # Reproject to Web Mercator for basemap display
    gdf_3857 = gdf.to_crs(epsg=3857)
    center_point = gpd.GeoSeries([Point(default_lon, default_lat)], crs="EPSG:4326").to_crs(epsg=3857)

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each disaster type with its own color
    for disaster_type, color in disaster_colors.items():
        group = gdf_3857[gdf_3857['disaster_type'] == disaster_type]
        if not group.empty:
            group.plot(ax=ax, color=color, marker='o', markersize=50, label=disaster_type)
            # Annotate each point with its disaster type
            for x, y in zip(group.geometry.x, group.geometry.y):
                ax.text(x + 100, y + 100, disaster_type, fontsize=8, color=color)
    
    # Mark hurricane if detected
    if hurricane:
        ax.scatter(center_point.geometry.x, center_point.geometry.y,
                   c='blue', s=200, marker='X', label='Hurricane Detected!')

    # Add basemap using an alternative provider with try/except
    try:
        ctx.add_basemap(ax, crs=gdf_3857.crs, source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        st.warning(f"Basemap could not be loaded: {e}")

    # Adjust the extents
    ax.set_xlim(gdf_3857.geometry.x.min() - 10000, gdf_3857.geometry.x.max() + 10000)
    ax.set_ylim(gdf_3857.geometry.y.min() - 10000, gdf_3857.geometry.y.max() + 10000)
    ax.set_title(f"Disaster Map for {target_date} at {target_hour}:00")
    ax.legend(loc='upper right')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# -----------------------------------------------------------------------------
# Severity Estimation Function
# -----------------------------------------------------------------------------
def estimate_severity(disaster_type, reading_value, weather_row):
    if disaster_type == 'Flood':
        return min(9, int((reading_value + weather_row['precipitation_mm']) / 20))
    elif disaster_type == 'Earthquake':
        return min(9, int(reading_value / 11))
    elif disaster_type == 'Fire':
        score = (reading_value - 45) + (25 - weather_row['humidity_%'])
        return min(9, max(1, int(score / 10)))
    elif disaster_type == 'Hurricane':
        wind = weather_row['wind_speed_kmph']
        pressure = weather_row['pressure_hPa']
        return min(9, int((wind - 100) / 10 + (1000 - pressure) / 5))
    return 1  


# -----------------------------------------------------------------------------
# Risk Prediction Functions and Hospital Proximity Check
# -----------------------------------------------------------------------------


from geopy.distance import geodesic
import joblib




# Load the model locally
cascade_model = joblib.load("/Users/Aprxx/Downloads/cascade_risk_model.pkl")

# Determine if near hospital
def is_near_hospital(lat, lon, hospitals, max_km=2):
    for _, hospital in hospitals.iterrows():
        if geodesic((lat, lon), (hospital.geometry.y, hospital.geometry.x)).km <= max_km:
            return 1
    return 0

# Predict risk
def predict_risk_from_sensor_row(sensor_row, weather_row, hospitals, model):
    
    sensor_type = sensor_row['sensor_type'].lower()
    reading_value = sensor_row['reading_value']
    lat = sensor_row['latitude']
    lon = sensor_row['longitude']

    # Determine disaster type based on sensor readings and weather conditions
    if sensor_type == 'flood' and (reading_value > 80 or (weather_row['precipitation_mm'] > 100 and weather_row['humidity_%'] > 85)):
        disaster_type = 'flood'
    elif sensor_type == 'seismic' and reading_value > 40:
        disaster_type = 'earthquake'
    elif sensor_type == 'temp' and reading_value > 45 and weather_row['humidity_%'] < 25:
        disaster_type = 'fire'
    elif weather_row['wind_speed_kmph'] > 100 and weather_row['pressure_hPa'] < 980:
        disaster_type = 'hurricane'
    else:
        return None, None, 0.0  # 无灾害，不计算风险

    # Estimate disaster severity
    severity = estimate_severity(disaster_type.title(), reading_value, weather_row)  # 注意：estimate_severity 要首字母大写
    
    # Check whether the sensor location is near a hospital
    near_hospital = is_near_hospital(lat, lon, hospitals)

    # 构造输入
    example = pd.DataFrame([{
        "disaster_type": disaster_type,
        "severity": severity,
        "near_hospital": near_hospital
    }])
    example = pd.get_dummies(example, columns=["disaster_type"])

    for col in model.feature_names_in_:
        if col not in example.columns:
            example[col] = 0
    example = example[model.feature_names_in_]

     
    risk_prob = model.predict_proba(example)[0][1]
    return severity, near_hospital, risk_prob
# -----------------------------------------------------------------------------
# OpenAI Integration for Emergency Action Advice
# -----------------------------------------------------------------------------
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=api_key)

def generate_action_advice(disaster_type, severity, hospital_risk):
    prompt = f"""
You are an emergency response expert.

Disaster type: {disaster_type}
Severity level: {severity} (on a scale from 1 to 9)
Estimated hospital overload risk: {hospital_risk:.2f} (0 to 1)

Please provide a short emergency action recommendation (2-3 sentences) suitable for city officials and emergency responders.
Be clear and concise.Very Concise!e.g., "Evacuate Zone C, reroute ambulances"Just tell the user the action plans. Mention key actions based on the risk level.
Respond in English.
"""

    response = client.chat.completions.create(
        model="gpt-4",  
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=120
    )

    return response.choices[0].message.content.strip()




# -----------------------------------------------------------------------------
# Page: Past Disasters
# -----------------------------------------------------------------------------
if page == "Past Disasters":
    st.title("Past Disasters Map -- Interactive Risk View")
    st.markdown("""
    <p style='font-size:16px; color:gray;'>
    Explore historical disaster events through our interactive map, where real-world data brings past incidents to life.
    Analyze patterns, assess risk, and gain insights that can help inform better future strategies.
    </p>
    """, unsafe_allow_html=True)
    
    # Aggregate historical disaster data by type, location, and date
    agg_by_type_loc_month = disaster_events.groupby(
        ['disaster_type', 'location', 'year', 'month']
    ).agg({
        'latitude': 'first',
        'longitude': 'first',
        'severity': 'mean',
        'casualties': 'mean',
        'economic_loss_million_usd': 'mean',
        'duration_hours': 'mean'
    }).reset_index()

    
    # Provide selection options for year and month using the available data
    available_years = sorted(disaster_events['year'].dropna().unique())
    available_months = sorted(disaster_events['month'].dropna().unique())
    selected_year = st.selectbox("Select Year", available_years)
    selected_month = st.selectbox("Select Month", available_months)

    # Filter aggregated data based on user's selections
    filtered_df = agg_by_type_loc_month[
        (agg_by_type_loc_month['year'] == selected_year) &
        (agg_by_type_loc_month['month'] == selected_month)
    ].copy()

    # Round numeric columns for display
    for col in ['severity', 'casualties', 'economic_loss_million_usd', 'duration_hours']:
        filtered_df[col] = filtered_df[col].round(2)

    # Prepare data for mapping
    layer_data = filtered_df.rename(columns={"longitude": "lon", "latitude": "lat"}).to_dict(orient="records")
    disaster_color_map = {
         "flood": [0, 0, 255, 255],
         "fire": [255, 0, 0, 255],
         "earthquake": [255, 165, 0, 255],
         "hurricane": [128, 0, 128, 255],
         "industrial accident": [0, 128, 0, 255]
    }
    
    # Add a new "color" attribute based on disaster_type
    for record in layer_data:
         # Ensure the key names match your dataset's casing; adjust if necessary.
         disaster_type = record.get("disaster_type", "")
         record["color"] = disaster_color_map.get(disaster_type, [0, 0, 0, 255])

    tooltip = {
        "html": """
        <b>Location:</b> {location}<br/>
        <b>Disaster Type:</b> {disaster_type}<br/>
        <b>Severity:</b> {severity}<br/>
        <b>Casualties:</b> {casualties}<br/>
        <b>Economic Loss ($M):</b> {economic_loss_million_usd}<br/>
        <b>Duration (hours):</b> {duration_hours}<br/>

        """,
        "style": {"backgroundColor": "white", "color": "black"}
    }

    disaster_layer = pdk.Layer(
        "ScatterplotLayer",
        data=layer_data,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=1000,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=filtered_df["latitude"].mean() if not filtered_df.empty else 37.7749,
        longitude=filtered_df["longitude"].mean() if not filtered_df.empty else -122.4194,
        zoom=8,
        pitch=0,
    )

    st.subheader(f"Past Risk Zones for {selected_year}-{selected_month:02d}")
    st.pydeck_chart(pdk.Deck(
        layers=[disaster_layer],
        initial_view_state=view_state,
        tooltip=tooltip
    ))


# -----------------------------------------------------------------------------
# Page: Future Predictions
# -----------------------------------------------------------------------------

elif page == "Future Predictions":
    st.title("Disaster Predictions Dashboard")
    st.markdown("""
    <p style='font-size:16px; color:gray;'>
    Gain insights into potential disasters using real-time sensor and weather data. 
    Our predictive analytics module helps city officials and emergency responders 
    prepare proactive response strategies. 
    </p>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### Disaster Type Legend
    - **Flood**: <span style="color:blue;">Blue</span>  
    - **Earthquake**: <span style="color:orange;">Orange</span>  
    - **Fire**: <span style="color:red;">Red</span>  
    - **Hurricane**: <span style="color:purple;">Purple</span>  
    """, unsafe_allow_html=True)

    # Form for users to select date and hour for predictions
    with st.form(key='future_predict_form'):
        selected_date = st.date_input("Select Date", value=pd.to_datetime("2023-01-01").date())
        selected_hour = st.slider("Select Hour", 0, 23, 12)
        submitted = st.form_submit_button("Generate Predictions")

    if submitted:
        # Filter sensor and weather data for the selected time
        sensor_sub = sensor_df[
            (sensor_df['date'] == selected_date) &
            (sensor_df['hour'] == selected_hour)
        ].copy()
        weather_sub = weather_df[
            (weather_df['date'] == selected_date) &
            (weather_df['hour'] == selected_hour)
        ]
        if sensor_sub.empty or weather_sub.empty:
            st.warning("Data for the selected time does not exist.")
        else:
            weather_row = weather_sub.iloc[0]

            # Classify disaster for each sensor reading using current weather data
            sensor_sub['disaster_type'] = sensor_sub.apply(lambda row: classify_disaster(row, weather_row), axis=1)
            sensor_disasters = sensor_sub[sensor_sub['disaster_type'].notnull()].copy()
            if sensor_disasters.empty:
                st.info("No disasters detected for the selected time.")
            else:
                sensor_disasters = sensor_disasters.reset_index(drop=True)
                sensor_disasters['label'] = sensor_disasters.index + 1

                predictions = []
                hospitals = infra_gdf[infra_gdf['type'] == 'hospital'].copy()
                for idx, row in sensor_disasters.iterrows():
                    sensor_series = pd.Series({
                        'sensor_type': row['sensor_type'],
                        'reading_value': row['reading_value'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude']
                    })
                    severity, near_hospital, risk_prob = predict_risk_from_sensor_row(sensor_series, weather_row, hospitals, cascade_model)
                    predictions.append({
                        'Label': row['label'],
                        'Disaster Type': row['disaster_type'],
                        'Sensor Type': row['sensor_type'],
                        'Reading Value': row['reading_value'],
                        'Latitude': row['latitude'],
                        'Longitude': row['longitude'],
                        'Predicted Severity': severity,
                        'Hospital Nearby': "Yes" if near_hospital else "No",
                        'Hospital Overload Risk Probability': f"{risk_prob:.2f}"
                    })
                preds_df = pd.DataFrame(predictions)

                gdf = gpd.GeoDataFrame(
                    sensor_disasters,
                    geometry=gpd.points_from_xy(sensor_disasters['longitude'], sensor_disasters['latitude']),
                    crs="EPSG:4326"
                )
                gdf_3857 = gdf.to_crs(epsg=3857)

                disaster_colors = {
                    "Flood": "red",
                    "Earthquake": "orange",
                    "Fire": "purple",
                    "Hurricane": "blue"
                }
                default_color = "black"

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_title(f"Disaster Map for {selected_date} at {selected_hour}:00")
                for dtype, group in gdf_3857.groupby("disaster_type"):
                    color = disaster_colors.get(dtype, default_color)
                    ax.scatter(group.geometry.x, group.geometry.y, s=50, color=color, marker='o')
                    for _, point in group.iterrows():
                        ax.text(point.geometry.x + 100, point.geometry.y + 100, f"{point['label']}", fontsize=10, color="black", weight="bold")

                try:
                    ctx.add_basemap(ax, crs=gdf_3857.crs, source=ctx.providers.OpenStreetMap.Mapnik)
                except Exception as e:
                    st.warning(f"Basemap could not be loaded: {e}")

                ax.set_axis_off()
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                st.image(buf, caption="Static Disaster Map with Labeled Points", use_column_width=True)

                st.markdown("### Prediction Details for Each Labeled Disaster")
                st.dataframe(preds_df)

                st.markdown("### Emergency Action Recommendations")
                st.write("Below are the generated emergency action recommendations for each disaster point:")
                for _, pred in preds_df.iterrows():
                    dt = pred["Disaster Type"]
                    sev = pred["Predicted Severity"]
                    risk = float(pred["Hospital Overload Risk Probability"])
                    with st.expander(f"Disaster Point {pred['Label']} ({dt})"):
                        try:
                            advice = generate_action_advice(dt, sev, risk)
                        except Exception as e:
                            advice = f"Error generating advice: {e}"
                        st.write(advice)


# -----------------------------------------------------------------------------
# Page: Social Media Detector
# -----------------------------------------------------------------------------

elif page == "Social Media Detector":
    st.title("Social Media Fake Post Detector")
    st.markdown("""
    Enter the details of a tweet below. The system checks for disaster-related keywords in your tweet and then verifies if there are any sensor readings around the provided time and location. If no sensor data supports a disaster in that context, or if the claimed disaster type does not match the sensor reading, the tweet is flagged as fake.
    """)

    # Set default date and hour
    tweet_date = st.date_input("Select Tweet Date", value=datetime(2023, 1, 1).date())
    tweet_hour = st.slider("Select Hour", min_value=0, max_value=23, value=12)

    # User input
    tweet_text = st.text_area("Tweet Text", help="Enter the tweet content that may mention a disaster.")
    tweet_lat = st.number_input("Tweet Latitude", value=0.0, format="%.6f")
    tweet_lon = st.number_input("Tweet Longitude", value=0.0, format="%.6f")
    check_button = st.button("Check Tweet")

    result_color = None

    if check_button:
        tweet_timestamp = datetime.combine(tweet_date, datetime.min.time()).replace(hour=tweet_hour)

        lower_text = tweet_text.lower()
        disaster_keywords = ["flood", "earthquake", "fire", "hurricane"]

        # Extract claimed disaster from tweet text
        claimed_disasters = [word for word in disaster_keywords if word in lower_text]
        if not claimed_disasters:
            st.info("Tweet does not mention any known disaster type.")
            result_color = "blue"
        else:
            claimed_disaster = claimed_disasters[0]  # Only check first detected keyword
            target_date = tweet_timestamp.date()
            target_hour = tweet_timestamp.hour

            sensor_subset = sensor_df[
                (sensor_df["date"] == target_date) &
                (sensor_df["hour"] == target_hour)
            ]
            weather_subset = weather_df[
                (weather_df["date"] == target_date) &
                (weather_df["hour"] == target_hour)
            ]
            if sensor_subset.empty or weather_subset.empty:
                st.warning("No sensor or weather data available for the selected time.")
                result_color = "blue"
            else:
                # Find nearby sensors
                sensor_subset["dist_km"] = sensor_subset.apply(
                    lambda row: geodesic((tweet_lat, tweet_lon), (row["latitude"], row["longitude"])).km, axis=1
                )
                nearby_sensors = sensor_subset[sensor_subset["dist_km"] <= 3]

                if nearby_sensors.empty:
                    st.success("Fake Tweet: No nearby sensor data confirms any disaster.")
                    result_color = "red"
                else:
                    weather_row = weather_subset.iloc[0]
                    nearby_sensors["disaster_type"] = nearby_sensors.apply(
                        lambda row: classify_disaster(row, weather_row), axis=1
                    )
                    detected_disasters = nearby_sensors["disaster_type"].dropna().str.lower().unique().tolist()

                    if claimed_disaster in detected_disasters:
                        st.success(f"Real Tweet: Detected {claimed_disaster} disaster in sensor data.")
                        result_color = "green"
                    elif detected_disasters:
                        st.success(f"Fake Tweet: Sensors show {', '.join(detected_disasters)}, not {claimed_disaster}.")
                        result_color = "red"
                    else:
                        st.success("Fake Tweet: No disaster detected by sensors.")
                        result_color = "red"
