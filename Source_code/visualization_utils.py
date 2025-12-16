"""
Module de visualisation et d'analyse :
- heatmap de corrélation
- scatter Distance vs fare_amount avec droite de régression
- analyse des résidus pour le meilleur modèle
- importance des features pour RandomForest
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


def plot_correlation_heatmap(df: pd.DataFrame):
    """Affiche une heatmap de corrélation des features."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()


def plot_distance_vs_fare(df: pd.DataFrame):
    """Scatter Distance vs fare_amount avec droite de régression."""
    sns.lmplot(
        x="Distance",
        y="fare_amount",
        data=df,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
    )
    plt.title("Distance vs Fare Amount (with Regression Line)")
    plt.show()


def plot_model_performance_bar(results_df: pd.DataFrame):
    """Barplot des R2 des différents modèles."""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="R2_Score",
        y="Model",
        data=results_df,
        hue="Model",
        palette="viridis",
        legend=False,
    )
    plt.title("Model Performance (Higher R² = Better)", fontsize=14)
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()


def plot_residuals_best_model(best_model, X_test_scaled, y_test, model_name: str):
    """Histogramme des résidus et résidus vs prédictions pour le meilleur modèle."""
    y_pred_best = best_model.predict(X_test_scaled)
    residuals = y_test - y_pred_best

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(residuals, bins=30, kde=True, color="steelblue")
    plt.title(f"Residual Distribution ({model_name})")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred_best, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residuals")

    plt.tight_layout()
    plt.show()


def plot_feature_importances(X_train_scaled, y_train, feature_names):
    """Apprend un RandomForest et affiche l'importance des features."""
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    importances = rf_model.feature_importances_
    feat_imp = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feat_imp.head(15),
        hue="Feature",
        palette="coolwarm",
        legend=False,
    )
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

    return feat_imp


if __name__ == "__main__":
    print("Ce module contient uniquement des fonctions de visualisation.")


