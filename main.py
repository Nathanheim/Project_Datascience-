"""
Main script for Uber fare prediction.

This file orchestrates:
- data loading and preprocessing
- sklearn model training
- (optional) PanelOLS model
- basic visualization
- tuning + saving the best RandomForest
"""

from pathlib import Path
import joblib
import sys

from Source_code.config import (
    DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH,
    ENABLE_VISUALIZATIONS,
    ENABLE_PANEL_OLS,
)
from Source_code.data_preprocessing import full_preprocessing_pipeline
from Source_code.modeling import (
    build_panel_ols,
    train_sklearn_models,
)
from Source_code.visualization_utils import (
    plot_correlation_heatmap,
    plot_distance_vs_fare,
    plot_model_performance_bar,
    plot_residuals_best_model,
    plot_feature_importances,
)
from Source_code.tuning_and_saving import tune_random_forest, evaluate_model, save_model


def main():
    """Main function with complete error handling."""
    try:
        # 1. Path to CSV (from config)
        csv_path = DATASET_PATH
        
        if not csv_path.exists():
            print(f"ERROR: File {csv_path} does not exist.")
            print("   Please ensure the dataset/uber.csv file exists.")
            sys.exit(1)

        # 2. Complete preprocessing
        print("Starting data preprocessing...")
        try:
            data = full_preprocessing_pipeline(str(csv_path))
        except Exception as e:
            print(f"ERROR during preprocessing: {str(e)}")
            print("   Please verify that the CSV file is valid and contains the correct columns.")
            sys.exit(1)
        
        df_clean = data["df_clean"]
        X = data["X"]
        y = data["y"]
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        X_train_scaled = data["X_train_scaled"]
        X_test_scaled = data["X_test_scaled"]
        scaler = data["scaler"]

        print("Preprocessing completed. Clean data shape:", df_clean.shape)
        
        if df_clean.shape[0] == 0:
            print("ERROR: No data after preprocessing.")
            sys.exit(1)

        # 3. (Optional) PanelOLS model
        if ENABLE_PANEL_OLS:
            panel_results = build_panel_ols(df_clean)
            print(panel_results["results"].summary)

        # 4. sklearn models
        print("\nTraining sklearn models...")
        try:
            results_df, trained_models = train_sklearn_models(X_train_scaled, X_test_scaled, y_train, y_test)
            print("\n=== sklearn Model Results ===")
            print(results_df)
        except Exception as e:
            print(f"ERROR during model training: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 5. Basic visualizations (if enabled)
        if ENABLE_VISUALIZATIONS:
            plot_correlation_heatmap(df_clean)
            plot_distance_vs_fare(df_clean)
            plot_model_performance_bar(results_df)

        # 6. Select best sklearn model (by R2)
        if results_df.empty:
            print("ERROR: No model results available.")
            sys.exit(1)
            
        best_model_name = results_df.iloc[0]["Model"]
        print(f"\nBest sklearn model (by R2): {best_model_name}")

        # Get the already trained best model (no duplication!)
        best_model = trained_models.get(best_model_name)
        if best_model is None:
            print(f"ERROR: Model '{best_model_name}' not found in trained models.")
            sys.exit(1)
        
        # Use the already trained model for visualizations (if enabled)
        if ENABLE_VISUALIZATIONS:
            plot_residuals_best_model(best_model, X_test_scaled, y_test, best_model_name)
            feat_imp = plot_feature_importances(X_train_scaled, y_train, X.columns)
            print("\nTop features by importance:")
            print(feat_imp.head())

        # 7. Tuning + saving the best RandomForest (on unscaled X_train)
        print("\nTuning RandomForest (this may take 15-20 minutes)...")
        try:
            best_rf = tune_random_forest(X_train, y_train)
        except KeyboardInterrupt:
            print("\nWARNING: Tuning interrupted by user (Ctrl+C).")
            print("   The model was not saved.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR during tuning: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        try:
            metrics = evaluate_model(best_rf, X_test, y_test)
        except Exception as e:
            print(f"ERROR during evaluation: {str(e)}")
            sys.exit(1)

        # 8. Save model (paths from config)
        try:
            save_model(best_rf, path=str(MODEL_PATH))
            print(f"\nRandomForest model saved in {MODEL_PATH}")
        except Exception as e:
            print(f"ERROR during model saving: {str(e)}")
            print("   The model was trained but could not be saved.")
            sys.exit(1)

        # Save scaler for reference (not used by RandomForest)
        try:
            joblib.dump(scaler, SCALER_PATH)
            print(f"Scaler saved in {SCALER_PATH}")
        except Exception as e:
            print(f"WARNING: Unable to save scaler: {str(e)}")
            # Non-blocking, continue

        print("\nPipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nWARNING: Script interrupted by user (Ctrl+C).")
        print("   Partially processed data was not saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
