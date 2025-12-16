"""
Module pour le tri et l'organisation des données
Contient les fonctions pour trier les résultats des modèles et les importances des features
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def trier_resultats_modeles(results_df, colonne_tri='R2_Score', ordre_decroissant=True):
    """
    Trie les résultats des modèles selon une colonne spécifiée
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame contenant les résultats des modèles (Model, R2_Score, MAE, RMSE)
    colonne_tri : str
        Nom de la colonne selon laquelle trier (par défaut: 'R2_Score')
    ordre_decroissant : bool
        Si True, tri décroissant (du meilleur au pire), sinon croissant
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame trié avec index réinitialisé
    """
    results_df_trie = results_df.sort_values(by=colonne_tri, ascending=not ordre_decroissant).reset_index(drop=True)
    return results_df_trie


def trier_importances_features(feature_names, importances, ordre_decroissant=True):
    """
    Crée un DataFrame avec les features et leurs importances, trié par importance
    
    Parameters:
    -----------
    feature_names : array-like
        Liste des noms des features
    importances : array-like
        Liste des valeurs d'importance correspondantes
    ordre_decroissant : bool
        Si True, tri décroissant (features les plus importantes en premier)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec colonnes 'Feature' et 'Importance', trié par importance
    """
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=not ordre_decroissant)
    
    return feat_imp


def preparer_donnees_panel(df, index_colonnes=['pickup_day', 'pickup_hour']):
    """
    Prépare les données pour une analyse panel en créant un index multi-niveaux
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    index_colonnes : list
        Liste des colonnes à utiliser comme index (par défaut: ['pickup_day', 'pickup_hour'])
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec index multi-niveaux configuré
    """
    df_panel = df.set_index(index_colonnes)
    return df_panel


def trier_par_colonne(df, colonne, ordre_decroissant=True):
    """
    Fonction générique pour trier un DataFrame par une colonne
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame à trier
    colonne : str
        Nom de la colonne selon laquelle trier
    ordre_decroissant : bool
        Si True, tri décroissant, sinon croissant
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame trié avec index réinitialisé
    """
    df_trie = df.sort_values(by=colonne, ascending=not ordre_decroissant).reset_index(drop=True)
    return df_trie


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple 1: Trier les résultats des modèles
    print("=== Exemple: Tri des résultats des modèles ===")
    exemple_results = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'R2_Score': [0.608, 0.813, 0.800],
        'MAE': [2.85, 2.03, 2.07],
        'RMSE': [5.89, 4.07, 4.21]
    })
    
    results_tries = trier_resultats_modeles(exemple_results, 'R2_Score', ordre_decroissant=True)
    print(results_tries)
    print()
    
    # Exemple 2: Trier les importances des features
    print("=== Exemple: Tri des importances des features ===")
    feature_names_ex = ['Distance', 'passenger_count', 'pickup_hour', 'pickup_month']
    importances_ex = [0.65, 0.15, 0.12, 0.08]
    
    feat_imp_trie = trier_importances_features(feature_names_ex, importances_ex)
    print(feat_imp_trie)
    print()
    
    # Exemple 3: Préparation des données panel
    print("=== Exemple: Préparation des données panel ===")
    exemple_df = pd.DataFrame({
        'pickup_day': [1, 1, 2, 2],
        'pickup_hour': [10, 14, 10, 14],
        'fare_amount': [15.5, 18.2, 16.1, 19.3],
        'Distance': [5.2, 6.8, 5.5, 7.1]
    })
    
    df_panel_ex = preparer_donnees_panel(exemple_df)
    print(df_panel_ex)
    print()

