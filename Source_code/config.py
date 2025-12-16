"""
Configuration centralisée pour le projet Uber Fare Prediction.

Toutes les valeurs de configuration sont centralisées ici pour faciliter
la maintenance et éviter la duplication de code.
"""

from pathlib import Path

# ============================================================================
# Chemins de fichiers
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "uber.csv"
MODEL_PATH = Path(__file__).parent / "uber_random_forest_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"

# ============================================================================
# Paramètres de randomisation
# ============================================================================
RANDOM_STATE = 42  # Pour reproductibilité

# ============================================================================
# Paramètres de preprocessing
# ============================================================================
TEST_SIZE = 0.2  # 20% pour le test set, 80% pour le train
MAX_DISTANCE = 60  # Distance maximale en km (outlier threshold)
MAX_FARE = 100  # Tarif maximal en $ (outlier threshold)
MIN_PASSENGERS = 1
MAX_PASSENGERS = 9

# ============================================================================
# Paramètres de tuning
# ============================================================================
N_ITER_RANDOM_SEARCH = 10  # Nombre d'itérations pour RandomizedSearchCV
N_SPLITS_CV = 3  # Nombre de folds pour validation croisée
N_JOBS = 2  # Nombre de jobs parallèles (réduit pour Mac)

# ============================================================================
# Paramètres des modèles
# ============================================================================
N_ESTIMATORS_RF = 100  # Nombre d'arbres pour Random Forest initial

# ============================================================================
# Options (flags)
# ============================================================================
ENABLE_VISUALIZATIONS = False  # Activer/désactiver les visualisations
ENABLE_PANEL_OLS = False  # Activer/désactiver PanelOLS
VERBOSE = 1  # Niveau de verbosité

