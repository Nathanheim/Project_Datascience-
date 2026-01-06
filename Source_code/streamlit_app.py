"""
Streamlit application for Uber fare prediction.

This web application allows:
- Enter GPS coordinates or distance directly
- Automatically calculate distance if coordinates are provided
- Make predictions with a modern and interactive interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go

# Import haversine function from preprocessing module
from data_preprocessing import haversine

# Configuration de la page
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load RandomForest model from saved file."""
    # New filename to avoid confusion with old scaler
    model_path = Path(__file__).parent / "uber_random_forest_model.pkl"
    
    if not model_path.exists():
        st.error(f"ERROR: File {model_path} does not exist.\n\n"
                "Please first run `python main.py` to train and save the model.")
        return None
    
    try:
        model = joblib.load(model_path)
        # Security: verify that loaded object has a predict method
        if not hasattr(model, "predict"):
            st.error("ERROR: Loaded model file does not contain a valid model (no 'predict' method).")
            return None
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"ERROR: Error loading model: {str(e)}")
        return None


def calculate_distance_from_coords(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """Calculate distance from GPS coordinates."""
    try:
        distance = haversine(pickup_lon, dropoff_lon, pickup_lat, dropoff_lat)
        return round(distance, 2)
    except Exception as e:
        return None


def validate_inputs(input_mode, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, 
                   distance_direct, passenger_count, year, month, day, hour):
    """Validate all input fields."""
    errors = []
    
    # Validation based on mode
    if input_mode == "coordinates":
        try:
            pickup_lat_val = float(pickup_lat)
            pickup_lon_val = float(pickup_lon)
            dropoff_lat_val = float(dropoff_lat)
            dropoff_lon_val = float(dropoff_lon)
            
            if not (-90 <= pickup_lat_val <= 90) or not (-90 <= dropoff_lat_val <= 90):
                errors.append("ERROR: Latitudes must be between -90 and 90")
            if not (-180 <= pickup_lon_val <= 180) or not (-180 <= dropoff_lon_val <= 180):
                errors.append("ERROR: Longitudes must be between -180 and 180")
            
            distance = calculate_distance_from_coords(
                pickup_lat_val, pickup_lon_val, dropoff_lat_val, dropoff_lon_val
            )
            if distance is None or distance <= 0:
                errors.append("ERROR: Invalid distance calculated from coordinates")
            elif distance > 60:
                errors.append(f"ERROR: Distance too large ({distance:.2f} km). Maximum: 60 km")
                
        except ValueError:
            errors.append("ERROR: Invalid GPS coordinates (must be numbers)")
    else:
        try:
            distance_val = float(distance_direct)
            if distance_val <= 0:
                errors.append("ERROR: Distance must be positive")
            elif distance_val > 60:
                errors.append(f"ERROR: Distance too large ({distance_val:.2f} km). Maximum: 60 km")
        except ValueError:
            errors.append("ERROR: Invalid distance (must be a number)")
    
    # Passenger count validation
    try:
        passenger_count_val = int(passenger_count)
        if passenger_count_val < 1 or passenger_count_val >= 10:
            errors.append("ERROR: Number of passengers must be between 1 and 9")
    except ValueError:
        errors.append("ERROR: Invalid passenger count")
    
    # Date/time validation
    try:
        year_val = int(year)
        month_val = int(month)
        day_val = int(day)
        hour_val = int(hour)
        
        if not (2000 <= year_val <= 2030):
            errors.append("ERROR: Year must be between 2000 and 2030")
        if not (1 <= month_val <= 12):
            errors.append("ERROR: Month must be between 1 and 12")
        if not (1 <= day_val <= 31):
            errors.append("ERROR: Day must be between 1 and 31")
        if not (0 <= hour_val <= 23):
            errors.append("ERROR: Hour must be between 0 and 23")
        
        # Calculate day of week
        try:
            date_obj = datetime(year_val, month_val, day_val)
            dayofweek = date_obj.weekday()  # 0 = Monday, 6 = Sunday
        except ValueError:
            errors.append("ERROR: Invalid date")
            
    except ValueError:
        errors.append("ERROR: Invalid date/time (must be integers)")
    
    return errors, dayofweek if "dayofweek" in locals() else None


def make_prediction(model, input_mode, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                   distance_direct, passenger_count, year, month, day, hour, dayofweek):
    """Perform fare prediction with all advanced features."""
    try:
        import numpy as np
        from modeling import (
            add_cyclical_features,
            add_traffic_features,
            add_geographical_features,
            add_interaction_features,
            add_time_based_features,
        )
        
        # Get distance
        if input_mode == "coordinates":
            distance = calculate_distance_from_coords(
                float(pickup_lat), float(pickup_lon),
                float(dropoff_lat), float(dropoff_lon)
            )
            pickup_lat_val = float(pickup_lat)
            pickup_lon_val = float(pickup_lon)
            dropoff_lat_val = float(dropoff_lat)
            dropoff_lon_val = float(dropoff_lon)
        else:
            distance = float(distance_direct)
            pickup_lat_val = 40.7128  # Default latitude (NYC)
            pickup_lon_val = -74.0060  # Default longitude (NYC)
            dropoff_lat_val = 40.7128
            dropoff_lon_val = -74.0060
        
        # Create input DataFrame with basic features
        input_data = pd.DataFrame({
            "pickup_latitude": [pickup_lat_val],
            "pickup_longitude": [pickup_lon_val],
            "dropoff_latitude": [dropoff_lat_val],
            "dropoff_longitude": [dropoff_lon_val],
            "passenger_count": [int(passenger_count)],
            "pickup_year": [int(year)],
            "pickup_month": [int(month)],
            "pickup_day": [int(day)],
            "pickup_hour": [int(hour)],
            "pickup_dayofweek": [dayofweek],
            "Distance": [distance],
        })
        
        # Add advanced features (in same order as pipeline)
        input_data = add_cyclical_features(input_data)
        input_data = add_traffic_features(input_data)
        input_data = add_geographical_features(input_data)
        input_data = add_interaction_features(input_data)
        input_data = add_time_based_features(input_data)
        
        # Remove location columns (as in pipeline)
        cols_to_drop = ["pickup_longitude", "dropoff_longitude"]
        existing = [c for c in cols_to_drop if c in input_data.columns]
        input_data = input_data.drop(columns=existing)
        
        # Prediction (RandomForest does not use scaler)
        fare_prediction = model.predict(input_data)[0]
        
        return fare_prediction, distance
        
    except Exception as e:
        st.error(f"ERROR: Error during prediction: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None, None


def main():
    """Main function of Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Uber Fare Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        ### How to use this application:
        
        1. **Choose input mode**:
           - GPS Coordinates: enter departure and arrival coordinates
           - Direct Distance: enter distance directly in km
        
        2. **Fill in trip information**
        
        3. **Click "Predict Fare"** to get the estimate
        
        ### Notes:
        - Maximum distance is 60 km
        - Number of passengers must be between 1 and 9
        - GPS coordinates are automatically calculated if provided
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model used")
        st.info("Random Forest Regressor\n\nTrained on a dataset of 200,000 Uber trips")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Data Input")
        
        # Input mode
        input_mode = st.radio(
            "**Input Mode:**",
            ["coordinates", "distance"],
            format_func=lambda x: "📍 GPS Coordinates" if x == "coordinates" else "📏 Direct Distance",
            help="Choose how you want to enter the distance"
        )
        
        st.markdown("---")
        
        # GPS Coordinates
        if input_mode == "coordinates":
            st.subheader("📍 GPS Coordinates")
            
            col_pickup, col_dropoff = st.columns(2)
            
            with col_pickup:
                st.markdown("**Pickup Point**")
                pickup_lat = st.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    value=40.7128,
                    step=0.0001,
                    format="%.6f",
                    key="pickup_lat"
                )
                pickup_lon = st.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    value=-74.0060,
                    step=0.0001,
                    format="%.6f",
                    key="pickup_lon"
                )
            
            with col_dropoff:
                st.markdown("**Dropoff Point**")
                dropoff_lat = st.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    value=40.7589,
                    step=0.0001,
                    format="%.6f",
                    key="dropoff_lat"
                )
                dropoff_lon = st.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    value=-73.9851,
                    step=0.0001,
                    format="%.6f",
                    key="dropoff_lon"
                )
            
            # Automatic distance calculation
            if all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
                calculated_distance = calculate_distance_from_coords(
                    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
                )
                if calculated_distance:
                    st.success(f"📏 Calculated distance: **{calculated_distance:.2f} km**")
                else:
                    st.warning("WARNING: Unable to calculate distance")
            
            distance_direct = None
        
        # Direct distance
        else:
            st.subheader("📏 Distance")
            distance_direct = st.number_input(
                "Distance (km)",
                min_value=0.1,
                max_value=60.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Maximum distance: 60 km"
            )
            pickup_lat = pickup_lon = dropoff_lat = dropoff_lon = None
        
        st.markdown("---")
        
        # Trip information
        st.subheader("🚕 Trip Information")
        
        col_passenger, col_date = st.columns(2)
        
        with col_passenger:
            passenger_count = st.number_input(
                "👥 Number of passengers",
                min_value=1,
                max_value=9,
                value=1,
                step=1,
                help="Between 1 and 9 passengers"
            )
        
        with col_date:
            # Date and time
            now = datetime.now()
            col_year, col_month, col_day, col_hour = st.columns(4)
            
            with col_year:
                year = st.number_input(
                    "Year",
                    min_value=2000,
                    max_value=2030,
                    value=now.year,
                    step=1,
                    key="year"
                )
            
            with col_month:
                month = st.number_input(
                    "Month",
                    min_value=1,
                    max_value=12,
                    value=now.month,
                    step=1,
                    key="month"
                )
            
            with col_day:
                day = st.number_input(
                    "Day",
                    min_value=1,
                    max_value=31,
                    value=now.day,
                    step=1,
                    key="day"
                )
            
            with col_hour:
                hour = st.number_input(
                    "Hour",
                    min_value=0,
                    max_value=23,
                    value=now.hour,
                    step=1,
                    key="hour"
                )
        
        # Prediction button
        st.markdown("---")
        predict_button = st.button(
            "🔮 Predict Fare",
            type="primary",
            use_container_width=True,
            help="Click to get fare prediction"
        )
    
    with col2:
        st.header("📊 Results")
        
        if predict_button:
            # Validation
            errors, dayofweek = validate_inputs(
                input_mode, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                distance_direct, passenger_count, year, month, day, hour
            )
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Prediction
                fare_prediction, distance_used = make_prediction(
                    model, input_mode, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                    distance_direct, passenger_count, year, month, day, hour, dayofweek
                )
                
                if fare_prediction:
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>💰 Predicted Fare</h2>
                        <div class="prediction-value">${fare_prediction:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional information
                    st.markdown("### 📋 Prediction Details")
                    
                    info_data = {
                        "Distance": [f"{distance_used:.2f} km"],
                        "Passengers": [f"{passenger_count}"],
                        "Date": [f"{int(day)}/{int(month)}/{int(year)}"],
                        "Hour": [f"{int(hour)}h"],
                    }
                    info_df = pd.DataFrame(info_data)
                    st.dataframe(info_df, use_container_width=True, hide_index=True)
                    
                    # Visualization chart (optional)
                    st.markdown("### 📈 Visualization")
                    
                    # Create simple chart with Plotly
                    fig = go.Figure()
                    
                    fig.add_trace(go.Indicator(
                        mode = "gauge+number",
                        value = fare_prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Predicted Fare ($)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgray"},
                                {'range': [20, 50], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Informative message
                    st.info("💡 This prediction is based on a machine learning model trained on historical Uber data.")
        else:
            st.info("👆 Fill in the information on the left and click 'Predict Fare' to get an estimate.")
            
            # Display example
            st.markdown("### 💡 Example")
            st.markdown("""
            **GPS Coordinates (New York):**
            - Departure: 40.7128, -74.0060 (Times Square)
            - Arrival: 40.7589, -73.9851 (Central Park)
            - Distance: ~5.5 km
            - Passengers: 1
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "🚗 Uber Fare Prediction Application | "
        "Based on a Random Forest Regressor model"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

