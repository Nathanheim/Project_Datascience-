# 🚖 Uber Fare Prediction Project

This repository contains two main components:

1. **Uber Fare Prediction Model** – A complete data analysis and machine learning pipeline for predicting Uber ride fares.
2. **Uber Fare Prediction GUI Application** – A simple Tkinter-based user interface that allows users to input ride details and get fare predictions using the trained machine-learning model.

---

# 📌 1. Uber Fare Prediction – Model Notebook

This notebook includes:

### ✅ **Data Cleaning & Preprocessing**
- Removing missing values  
- Handling incorrect or unrealistic values  
- Feature engineering (distance, time features, etc.)

### ✅ **Exploratory Data Analysis (EDA)**
- Distribution plots  
- Outlier detection  
- Correlation heatmaps  
- Feature importance insights  

### ✅ **Baseline Model: Fixed-Effects Panel Regression**
As requested by the client, a classical econometric model (Fixed Effects Panel Regression) is included for benchmarking.

### ✅ **Machine Learning Models**
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Hyperparameter tuning using RandomizedSearchCV  
- Final best model saved as `.pkl`

### ✅ **Model Comparison**
A formal comparison between:
- Fixed-Effects Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

Using:
- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

**Final Best Model:**  
`RandomForestRegressor` with:
- R² ≈ **0.82**
- MAE ≈ **1.95**
- RMSE ≈ **3.99**

---

# 📌 2. Uber Fare Prediction – Tkinter GUI App

A simple, user-friendly GUI that loads the trained Random Forest model (`random_forest_model.pkl`) and predicts fares.

### Features:
- Clean graphical interface  
- Background image (`uber.jpg`)  
- User input fields:
  - Pickup/Dropoff Latitude  
  - Passenger Count  
  - Pickup Year  
  - Pickup Month  
  - Pickup Day  
  - Pickup Hour  
  - Pickup Day of Week  
  - Distance  

### Input Validation:
- Passenger Count → 1 to 6  
- Year → 2025 to 2030  
- Month → 1 to 12  
- Day → 1 to 30  
- Hour → 1 to 24  
- Day of Week → 1 to 7  
- Distance → must be > 1  

### Prediction:
Shows a popup message with the predicted fare:

> **Predicted Fare Amount: $X.XX**

### GUI Preview Screenshot:



![Uber GUI Screenshot](images/test.jpg)


---

# 🚀 How to Run the Project

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/uber-fare-prediction.git
cd Uber_fare_prediction
````

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

Recommended libraries:

* pandas
* numpy
* scikit-learn
* seaborn
* matplotlib
* linearmodels
* tkinter
* pillow
* joblib


### **3. Run the GUI Application**

Steps:

- Make sure the following files are in the same folder:

- app.py (your Tkinter code)

- random_forest_model.pkl

- uber.jpg (background image)

Open Command Prompt (CMD) inside the folder.
---
Run the app:
```
python app.py
```
# 🧠 Technologies Used

* Python
* Scikit-Learn
* pandas
* numpy
* seaborn
* matplotlib
* linearmodels (Fixed-Effects Regression)
* tkinter (GUI)
* pillow (Image handling)
* joblib (Model saving)

---

# 📊 Key Results

| Model                    | R²       | MAE      | RMSE     |
| ------------------------ | -------- | -------- | -------- |
| **Random Forest (best)** | **0.82** | **1.95** | **3.99** |
| Gradient Boosting        | 0.80     | 2.07     | 4.20     |
| Decision Tree            | 0.61     | 2.85     | 5.88     |
| Fixed-Effects Regression | Lower    | Higher   | Higher   |

# **Conclusion:**
➡️ Machine-Learning models significantly outperform the classical Fixed-Effects baseline.

➡️ Random Forest provides the best predictive accuracy.

---

# 🙌 Author

**Nathan Heim**

---

