# Uber Fare Prediction Project

A complete machine learning pipeline for predicting Uber ride fares, with a modern Streamlit web application interface.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Code Architecture](#code-architecture)
- [Author](#author)

---

## Overview

This project implements a comprehensive machine learning pipeline to predict Uber ride fares based on:
- Pickup and dropoff locations (GPS coordinates)
- Trip distance
- Passenger count
- Date and time features
- Temporal patterns (hour, day, month, day of week)

The project includes:
1. **Complete ML Pipeline** – Data preprocessing, feature engineering, model training, and evaluation
2. **Streamlit Web Application** – Interactive interface for making fare predictions
3. **Multiple ML Models** – Comparison of various algorithms including Random Forest, Gradient Boosting, and more

---

## 📁 Project Structure

```
Uber_fare_prediction-main/
├── main.py                      # Main pipeline orchestrator
├── Source_code/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── modeling.py              # ML models and feature improvements
│   ├── tuning_and_saving.py    # Hyperparameter tuning and model persistence
│   ├── visualization_utils.py  # Plotting functions
│   ├── streamlit_app.py         # Streamlit web application
│   ├── config.py                # Centralized configuration
│   ├── uber_random_forest_model.pkl  # Trained model (if generated)
│   └── scaler.pkl               # Saved scaler (if generated)
├── dataset/
│   └── uber.csv                 # Training data (200,000+ records)
├── results/                     # Visualization outputs
├── requirements.txt             # Python dependencies
├── README.md                    # This file
```

---

## Features

### Data Preprocessing
- Missing value handling (removal of rows with missing values)
- Outlier removal (distance ≤ 60 km, fare < $100, passenger count between 1 and 9)
- Feature engineering:
  - Haversine distance calculation (geodesic distance between two GPS points)
  - Temporal features (year, month, day, hour, day of week)
  - Cyclical features for temporal patterns (sine/cosine encoding)
  - Traffic-related features (rush hours, weekends, night time)
  - Interaction features (distance × passengers, distance²)
- Feature scaling with StandardScaler


### Machine Learning Models
- Linear Regression
- Ridge & Lasso Regression
- Decision Tree
- **Random Forest** (Best performing model)
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors
- Support Vector Regressor
- PanelOLS (Fixed-Effects Panel Regression) - for benchmarking

### Hyperparameter Optimization
- RandomizedSearchCV with 3-fold cross-validation
- Optimization of key Random Forest parameters:
  - Number of trees (n_estimators)
  - Maximum depth (max_depth)
  - Minimum sample thresholds (min_samples_split, min_samples_leaf)

### Web Application (Streamlit)
- Modern, interactive web interface
- Two input modes:
  - GPS coordinates (automatic distance calculation)
  - Direct distance input
- Real-time fare prediction
- Input validation
- Visualizations (gauges, charts)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Nathanheim/Uber_fare_prediction.git
cd Uber_fare_prediction-main
```

### 2. Create a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python projects.

**Using venv (built-in):**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Using uv (modern/fast):**
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Note:** If you skip the virtual environment, you can install directly:
```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- plotly
- joblib
- statsmodels
- linearmodels

---

## Usage

### Step 1: Train the Model

First, train and save the machine learning model:

```bash
python3 main.py
```

This will:
- Load and preprocess the dataset
- Train multiple ML models
- Perform hyperparameter tuning on Random Forest
- Save the best model as `uber_random_forest_model.pkl`
- Save the scaler as `scaler.pkl`

**Note:** The training process takes approximately 1 hour to execute, especially due to hyperparameter tuning.

### Step 2: Launch the Streamlit Application

Once the model is trained, launch the web application:

```bash
cd Source_code
python3 -m streamlit run streamlit_app.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

### Using the Application

1. **Choose input mode:**
   - **GPS Coordinates**: Enter pickup and dropoff coordinates (distance calculated automatically)
   - **Direct Distance**: Enter distance directly in km

2. **Fill in trip details:**
   - Number of passengers (1-9)
   - Date and time (year, month, day, hour)

3. **Click "Prédire le Tarif"** to get the fare prediction

---

## 📊 Model Performance

### Best Model: Random Forest Regressor

| Metric | Value |
|--------|-------|
| **R² Score** | **0.82** |
| **MAE** | **1.95** |
| **RMSE** | **3.99** |

### Model Comparison

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| **Random Forest** (best) | **0.82** | **1.95** | **3.99** |
| Gradient Boosting | 0.80 | 2.07 | 4.20 |
| Decision Tree | 0.61 | 2.85 | 5.88 |
| Linear Regression | ~0.70 | ~2.50 | ~4.50 |

**Conclusion:**
- Machine Learning models significantly outperform classical econometric models
- Random Forest provides the best predictive accuracy with R² ≈ 0.82
- The model achieves low error rates (MAE < $2)

---

## Technologies Used

- **Python 3.9+** – Main programming language
- **Scikit-Learn** – Machine learning algorithms and tools
- **pandas & numpy** – Data manipulation and analysis
- **Streamlit** – Web application framework
- **Plotly** – Interactive visualizations
- **matplotlib & seaborn** – Static visualizations
- **linearmodels** – Fixed-Effects Panel Regression
- **joblib** – Model serialization
- **statsmodels** – Statistical modeling

---

## Code Architecture

The project follows a modular architecture:

- **`config.py`** – Centralized configuration (paths, parameters, hyperparameters)
- **`data_preprocessing.py`** – Handles data loading, cleaning, and feature engineering
- **`modeling.py`** – Defines ML models and feature improvement functions
- **`tuning_and_saving.py`** – Hyperparameter tuning and model persistence
- **`visualization_utils.py`** – Plotting and visualization functions
- **`main.py`** – Orchestrates the entire pipeline
- **`streamlit_app.py`** – Web application interface

### Processing Pipeline

1. **Data Loading**: Read CSV file
2. **Cleaning**: Remove missing values and outliers
3. **Feature Engineering**: Create temporal features and calculate Haversine distance
4. **Preparation**: Separate features/target, train/test split, scaling
5. **Training**: Compare multiple ML models
6. **Optimization**: Tune hyperparameters of the best model (Random Forest)
7. **Saving**: Persist model and scaler with joblib




## Important Notes

- The trained model files (`.pkl`) are excluded from Git (see `.gitignore`) as they exceed GitHub's file size limits.
- To use the application, you must first train the model by running `main.py`.
- The model uses 3-fold cross-validation for hyperparameter optimization (configurable in `config.py`).
- The model is saved with joblib (recommended method for scikit-learn).

---


**Nathan Heimendinger-Dreyfus**

GitHub: [@Nathanheim](https://github.com/Nathanheim)


