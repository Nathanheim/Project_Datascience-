"""
Application Streamlit pour la prédiction du tarif Uber.

Cette application web permet de :
- Saisir les coordonnées GPS ou la distance directement
- Calculer automatiquement la distance si les coordonnées sont fournies
- Faire des prédictions avec une interface moderne et interactive
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go

# Import de la fonction haversine depuis le module de preprocessing
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
    """Charge le modèle RandomForest depuis le fichier sauvegardé."""
    # Nouveau nom de fichier pour éviter toute confusion avec un ancien scaler
    model_path = Path(__file__).parent / "uber_random_forest_model.pkl"
    
    if not model_path.exists():
        st.error(f"ERREUR : Le fichier {model_path} n'existe pas.\n\n"
                "Veuillez d'abord exécuter `python main.py` pour entraîner et sauvegarder le modèle.")
        return None
    
    try:
        model = joblib.load(model_path)
        # Sécurité : vérifier que l'objet chargé a bien une méthode predict
        if not hasattr(model, "predict"):
            st.error("ERREUR : Le fichier de modèle chargé ne contient pas un modèle valide (pas de méthode 'predict').")
            return None
        st.success("Modèle chargé avec succès!")
        return model
    except Exception as e:
        st.error(f"ERREUR : Erreur lors du chargement du modèle : {str(e)}")
        return None


def calculate_distance_from_coords(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """Calcule la distance à partir des coordonnées GPS."""
    try:
        distance = haversine(pickup_lon, dropoff_lon, pickup_lat, dropoff_lat)
        return round(distance, 2)
    except Exception as e:
        return None


def validate_inputs(input_mode, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, 
                   distance_direct, passenger_count, year, month, day, hour):
    """Valide tous les champs de saisie."""
    errors = []
    
    # Validation selon le mode
    if input_mode == "coordinates":
        try:
            pickup_lat_val = float(pickup_lat)
            pickup_lon_val = float(pickup_lon)
            dropoff_lat_val = float(dropoff_lat)
            dropoff_lon_val = float(dropoff_lon)
            
            if not (-90 <= pickup_lat_val <= 90) or not (-90 <= dropoff_lat_val <= 90):
                errors.append("ERREUR : Les latitudes doivent être entre -90 et 90")
            if not (-180 <= pickup_lon_val <= 180) or not (-180 <= dropoff_lon_val <= 180):
                errors.append("ERREUR : Les longitudes doivent être entre -180 et 180")
            
            distance = calculate_distance_from_coords(
                pickup_lat_val, pickup_lon_val, dropoff_lat_val, dropoff_lon_val
            )
            if distance is None or distance <= 0:
                errors.append("ERREUR : Distance invalide calculée à partir des coordonnées")
            elif distance > 60:
                errors.append(f"ERREUR : Distance trop grande ({distance:.2f} km). Maximum: 60 km")
                
        except ValueError:
            errors.append("ERREUR : Coordonnées GPS invalides (doivent être des nombres)")
    else:
        try:
            distance_val = float(distance_direct)
            if distance_val <= 0:
                errors.append("ERREUR : La distance doit être positive")
            elif distance_val > 60:
                errors.append(f"ERREUR : Distance trop grande ({distance_val:.2f} km). Maximum: 60 km")
        except ValueError:
            errors.append("ERREUR : Distance invalide (doit être un nombre)")
    
    # Validation du nombre de passagers
    try:
        passenger_count_val = int(passenger_count)
        if passenger_count_val < 1 or passenger_count_val >= 10:
            errors.append("ERREUR : Le nombre de passagers doit être entre 1 et 9")
    except ValueError:
        errors.append("ERREUR : Nombre de passagers invalide")
    
    # Validation de la date/heure
    try:
        year_val = int(year)
        month_val = int(month)
        day_val = int(day)
        hour_val = int(hour)
        
        if not (2000 <= year_val <= 2030):
            errors.append("ERREUR : L'année doit être entre 2000 et 2030")
        if not (1 <= month_val <= 12):
            errors.append("ERREUR : Le mois doit être entre 1 et 12")
        if not (1 <= day_val <= 31):
            errors.append("ERREUR : Le jour doit être entre 1 et 31")
        if not (0 <= hour_val <= 23):
            errors.append("ERREUR : L'heure doit être entre 0 et 23")
        
        # Calculer le jour de la semaine
        try:
            date_obj = datetime(year_val, month_val, day_val)
            dayofweek = date_obj.weekday()  # 0 = lundi, 6 = dimanche
        except ValueError:
            errors.append("ERREUR : Date invalide")
            
    except ValueError:
        errors.append("ERREUR : Date/heure invalides (doivent être des nombres entiers)")
    
    return errors, dayofweek if "dayofweek" in locals() else None


def make_prediction(model, input_mode, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                   distance_direct, passenger_count, year, month, day, hour, dayofweek):
    """Effectue la prédiction du tarif."""
    try:
        # Récupération de la distance
        if input_mode == "coordinates":
            distance = calculate_distance_from_coords(
                float(pickup_lat), float(pickup_lon),
                float(dropoff_lat), float(dropoff_lon)
            )
            pickup_lat_val = float(pickup_lat)
            dropoff_lat_val = float(dropoff_lat)
        else:
            distance = float(distance_direct)
            pickup_lat_val = 40.7128  # Latitude par défaut (NYC)
            dropoff_lat_val = 40.7128
        
        # Création du DataFrame d'entrée
        input_data = pd.DataFrame({
            "pickup_latitude": [pickup_lat_val],
            "dropoff_latitude": [dropoff_lat_val],
            "passenger_count": [int(passenger_count)],
            "pickup_year": [int(year)],
            "pickup_month": [int(month)],
            "pickup_day": [int(day)],
            "pickup_hour": [int(hour)],
            "pickup_dayofweek": [dayofweek],
            "Distance": [distance],
        })
        
        # Prédiction (RandomForest n'utilise pas de scaler)
        fare_prediction = model.predict(input_data)[0]
        
        return fare_prediction, distance
        
    except Exception as e:
        st.error(f"ERREUR : Erreur lors de la prédiction : {str(e)}")
        return None, None


def main():
    """Fonction principale de l'application Streamlit."""
    
    # En-tête
    st.markdown('<h1 class="main-header">🚗 Prédiction du Tarif Uber</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Chargement du modèle
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("ℹ️ Informations")
        st.markdown("""
        ### Comment utiliser cette application :
        
        1. **Choisissez le mode de saisie** :
           - Coordonnées GPS : entrez les coordonnées de départ et d'arrivée
           - Distance directe : entrez directement la distance en km
        
        2. **Remplissez les informations** du trajet
        
        3. **Cliquez sur "Prédire le Tarif"** pour obtenir l'estimation
        
        ### Notes :
        - La distance maximale est de 60 km
        - Le nombre de passagers doit être entre 1 et 9
        - Les coordonnées GPS sont calculées automatiquement si fournies
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Modèle utilisé")
        st.info("Random Forest Regressor\n\nEntraîné sur un dataset de 200 000 trajets Uber")
    
    # Contenu principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Saisie des données")
        
        # Mode de saisie
        input_mode = st.radio(
            "**Mode de saisie :**",
            ["coordinates", "distance"],
            format_func=lambda x: "📍 Coordonnées GPS" if x == "coordinates" else "📏 Distance directe",
            help="Choisissez comment vous voulez saisir la distance"
        )
        
        st.markdown("---")
        
        # Coordonnées GPS
        if input_mode == "coordinates":
            st.subheader("📍 Coordonnées GPS")
            
            col_pickup, col_dropoff = st.columns(2)
            
            with col_pickup:
                st.markdown("**Point de départ (Pickup)**")
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
                st.markdown("**Point d'arrivée (Dropoff)**")
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
            
            # Calcul automatique de la distance
            if all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
                calculated_distance = calculate_distance_from_coords(
                    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
                )
                if calculated_distance:
                    st.success(f"📏 Distance calculée : **{calculated_distance:.2f} km**")
                else:
                    st.warning("AVERTISSEMENT : Impossible de calculer la distance")
            
            distance_direct = None
        
        # Distance directe
        else:
            st.subheader("📏 Distance")
            distance_direct = st.number_input(
                "Distance (km)",
                min_value=0.1,
                max_value=60.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Distance maximale : 60 km"
            )
            pickup_lat = pickup_lon = dropoff_lat = dropoff_lon = None
        
        st.markdown("---")
        
        # Informations du trajet
        st.subheader("🚕 Informations du trajet")
        
        col_passenger, col_date = st.columns(2)
        
        with col_passenger:
            passenger_count = st.number_input(
                "👥 Nombre de passagers",
                min_value=1,
                max_value=9,
                value=1,
                step=1,
                help="Entre 1 et 9 passagers"
            )
        
        with col_date:
            # Date et heure
            now = datetime.now()
            col_year, col_month, col_day, col_hour = st.columns(4)
            
            with col_year:
                year = st.number_input(
                    "Année",
                    min_value=2000,
                    max_value=2030,
                    value=now.year,
                    step=1,
                    key="year"
                )
            
            with col_month:
                month = st.number_input(
                    "Mois",
                    min_value=1,
                    max_value=12,
                    value=now.month,
                    step=1,
                    key="month"
                )
            
            with col_day:
                day = st.number_input(
                    "Jour",
                    min_value=1,
                    max_value=31,
                    value=now.day,
                    step=1,
                    key="day"
                )
            
            with col_hour:
                hour = st.number_input(
                    "Heure",
                    min_value=0,
                    max_value=23,
                    value=now.hour,
                    step=1,
                    key="hour"
                )
        
        # Bouton de prédiction
        st.markdown("---")
        predict_button = st.button(
            "🔮 Prédire le Tarif",
            type="primary",
            use_container_width=True,
            help="Cliquez pour obtenir la prédiction du tarif"
        )
    
    with col2:
        st.header("📊 Résultats")
        
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
                # Prédiction
                fare_prediction, distance_used = make_prediction(
                    model, input_mode, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                    distance_direct, passenger_count, year, month, day, hour, dayofweek
                )
                
                if fare_prediction:
                    # Affichage du résultat
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>💰 Tarif Prédit</h2>
                        <div class="prediction-value">${fare_prediction:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Informations supplémentaires
                    st.markdown("### 📋 Détails de la prédiction")
                    
                    info_data = {
                        "Distance": [f"{distance_used:.2f} km"],
                        "Passagers": [f"{passenger_count}"],
                        "Date": [f"{int(day)}/{int(month)}/{int(year)}"],
                        "Heure": [f"{int(hour)}h"],
                    }
                    info_df = pd.DataFrame(info_data)
                    st.dataframe(info_df, use_container_width=True, hide_index=True)
                    
                    # Graphique de visualisation (optionnel)
                    st.markdown("### 📈 Visualisation")
                    
                    # Créer un graphique simple avec Plotly
                    fig = go.Figure()
                    
                    fig.add_trace(go.Indicator(
                        mode = "gauge+number",
                        value = fare_prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Tarif Prédit ($)"},
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
                    
                    # Message informatif
                    st.info("💡 Cette prédiction est basée sur un modèle de machine learning entraîné sur des données historiques Uber.")
        else:
            st.info("👆 Remplissez les informations à gauche et cliquez sur 'Prédire le Tarif' pour obtenir une estimation.")
            
            # Afficher un exemple
            st.markdown("### 💡 Exemple")
            st.markdown("""
            **Coordonnées GPS (New York) :**
            - Départ : 40.7128, -74.0060 (Times Square)
            - Arrivée : 40.7589, -73.9851 (Central Park)
            - Distance : ~5.5 km
            - Passagers : 1
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "🚗 Application de Prédiction du Tarif Uber | "
        "Basée sur un modèle Random Forest Regressor"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

