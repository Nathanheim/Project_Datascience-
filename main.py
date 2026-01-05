"""
Script principal pour la prédiction du tarif Uber.

Ce fichier orchestre :
- le chargement et le prétraitement des données
- l'entraînement des modèles sklearn
- (optionnel) le modèle PanelOLS
- la visualisation de base
- le tuning + sauvegarde du meilleur RandomForest
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
    """Fonction principale avec gestion d'erreurs complète."""
    try:
        # 1. Chemin vers le CSV (depuis config)
        csv_path = DATASET_PATH
        
        if not csv_path.exists():
            print(f"ERREUR : Le fichier {csv_path} n'existe pas.")
            print("   Vérifiez que le fichier dataset/uber.csv existe.")
            sys.exit(1)

        # 2. Prétraitement complet
        print("Début du prétraitement des données...")
        try:
            data = full_preprocessing_pipeline(str(csv_path))
        except Exception as e:
            print(f"ERREUR lors du prétraitement : {str(e)}")
            print("   Vérifiez que le fichier CSV est valide et contient les bonnes colonnes.")
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

        print("Prétraitement terminé. Shape données nettoyées :", df_clean.shape)
        
        if df_clean.shape[0] == 0:
            print("ERREUR : Aucune donnée après le prétraitement.")
            sys.exit(1)

        # 3. (Optionnel) Modèle PanelOLS
        if ENABLE_PANEL_OLS:
            panel_results = build_panel_ols(df_clean)
            print(panel_results["results"].summary)

        # 4. Modèles sklearn
        print("\nEntraînement des modèles sklearn...")
        try:
            results_df, trained_models = train_sklearn_models(X_train_scaled, X_test_scaled, y_train, y_test)
            print("\n=== Résultats des modèles sklearn ===")
            print(results_df)
        except Exception as e:
            print(f"ERREUR lors de l'entraînement des modèles : {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 5. Visualisations de base (si activées)
        if ENABLE_VISUALIZATIONS:
            plot_correlation_heatmap(df_clean)
            plot_distance_vs_fare(df_clean)
            plot_model_performance_bar(results_df)

        # 6. Sélection du meilleur modèle sklearn (par R2)
        if results_df.empty:
            print("ERREUR : Aucun résultat de modèle disponible.")
            sys.exit(1)
            
        best_model_name = results_df.iloc[0]["Model"]
        print(f"\nMeilleur modèle sklearn (d'après R2) : {best_model_name}")

        # Récupérer le meilleur modèle déjà entraîné (plus de duplication !)
        best_model = trained_models.get(best_model_name)
        if best_model is None:
            print(f"ERREUR : Modèle '{best_model_name}' non trouvé dans les modèles entraînés.")
            sys.exit(1)
        
        # Utiliser le modèle déjà entraîné pour les visualisations (si activées)
        if ENABLE_VISUALIZATIONS:
            plot_residuals_best_model(best_model, X_test_scaled, y_test, best_model_name)
            feat_imp = plot_feature_importances(X_train_scaled, y_train, X.columns)
            print("\nTop features par importance :")
            print(feat_imp.head())

        # 7. Tuning + sauvegarde du meilleur RandomForest (sur X_train non-scalé)
        print("\nTuning du RandomForest (cela peut prendre 15-20 minutes)...")
        try:
            best_rf = tune_random_forest(X_train, y_train)
        except KeyboardInterrupt:
            print("\nAVERTISSEMENT : Tuning interrompu par l'utilisateur (Ctrl+C).")
            print("   Le modèle n'a pas été sauvegardé.")
            sys.exit(1)
        except Exception as e:
            print(f"ERREUR lors du tuning : {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        try:
            metrics = evaluate_model(best_rf, X_test, y_test)
        except Exception as e:
            print(f"ERREUR lors de l'évaluation : {str(e)}")
            sys.exit(1)

        # 8. Sauvegarde du modèle (chemins depuis config)
        try:
            save_model(best_rf, path=str(MODEL_PATH))
            print(f"\nModèle RandomForest sauvegardé dans {MODEL_PATH}")
        except Exception as e:
            print(f"ERREUR lors de la sauvegarde du modèle : {str(e)}")
            print("   Le modèle a été entraîné mais n'a pas pu être sauvegardé.")
            sys.exit(1)

        # Sauvegarde du scaler pour référence (non utilisé par RandomForest)
        try:
            joblib.dump(scaler, SCALER_PATH)
            print(f"Scaler sauvegardé dans {SCALER_PATH}")
        except Exception as e:
            print(f"AVERTISSEMENT : Impossible de sauvegarder le scaler : {str(e)}")
            # Non bloquant, on continue

        print("\nPipeline terminé avec succès !")
        
    except KeyboardInterrupt:
        print("\n\nAVERTISSEMENT : Script interrompu par l'utilisateur (Ctrl+C).")
        print("   Les données partiellement traitées n'ont pas été sauvegardées.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERREUR inattendue : {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

