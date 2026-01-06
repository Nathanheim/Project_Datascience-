"""
Centralized configuration for Uber Fare Prediction project.

All configuration values are centralized here to facilitate
maintenance and avoid code duplication.
"""

from pathlib import Path

# ============================================================================
# File paths
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "uber.csv"
MODEL_PATH = Path(__file__).parent / "uber_random_forest_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"

# ============================================================================
# Randomization parameters
# ============================================================================
RANDOM_STATE = 42  # For reproducibility

# ============================================================================
# Preprocessing parameters
# ============================================================================
TEST_SIZE = 0.2  # 20% for test set, 80% for train
MAX_DISTANCE = 60  # Maximum distance in km (outlier threshold)
MAX_FARE = 100  # Maximum fare in $ (outlier threshold)
MIN_PASSENGERS = 1
MAX_PASSENGERS = 9

# ============================================================================
# Tuning parameters
# ============================================================================
N_ITER_RANDOM_SEARCH = 10  # Number of iterations for RandomizedSearchCV
N_SPLITS_CV = 3  # Number of folds for cross-validation
N_JOBS = 2  # Number of parallel jobs (reduced for Mac)

# ============================================================================
# Model parameters
# ============================================================================
N_ESTIMATORS_RF = 100  # Number of trees for initial Random Forest

# ============================================================================
# Options (flags)
# ============================================================================
ENABLE_VISUALIZATIONS = False  # Enable/disable visualizations
ENABLE_PANEL_OLS = False  # Enable/disable PanelOLS
VERBOSE = 1  # Verbosity level

