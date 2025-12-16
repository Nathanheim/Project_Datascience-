"""
Application graphique pour la prédiction du tarif Uber.

Cette application utilise le modèle RandomForest entraîné et permet de :
- Saisir les coordonnées GPS ou la distance directement
- Calculer automatiquement la distance si les coordonnées sont fournies
- Faire des prédictions avec une interface moderne
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# Import de la fonction haversine depuis le module de preprocessing
from data_preprocessing import haversine


class UberFarePredictorApp:
    """Application principale pour la prédiction du tarif Uber."""

    def __init__(self, root):
        self.root = root
        self.root.title("Uber Fare Prediction - Application")
        self.root.geometry("700x750")
        self.root.configure(bg="#f0f0f0")

        # Chargement du modèle et du scaler
        self.model = None
        self.scaler = None
        self.load_model_and_scaler()

        # Variables pour le mode de saisie
        self.input_mode = tk.StringVar(value="coordinates")  # "coordinates" ou "distance"

        # Création de l'interface
        self.create_widgets()

    def load_model_and_scaler(self):
        """Charge le modèle et le scaler depuis les fichiers sauvegardés."""
        try:
            model_path = Path(__file__).parent / "random_forest_model.pkl"
            scaler_path = Path(__file__).parent / "scaler.pkl"

            if model_path.exists():
                self.model = joblib.load(model_path)
                print(f"Modèle chargé depuis {model_path}")
            else:
                messagebox.showwarning(
                    "Modèle non trouvé",
                    f"Le fichier {model_path} n'existe pas.\n"
                    "Veuillez d'abord exécuter main.py pour entraîner et sauvegarder le modèle.",
                )
                return

            # Note: Le RandomForest n'utilise pas de scaler (modèle basé sur arbres)
            # Le scaler est sauvegardé pour référence mais n'est pas utilisé ici
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler chargé depuis {scaler_path} (non utilisé pour RandomForest)")
        except Exception as e:
            messagebox.showerror("Erreur de chargement", f"Erreur lors du chargement : {str(e)}")

    def create_widgets(self):
        """Crée tous les widgets de l'interface."""
        # Titre
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🚗 Prédiction du Tarif Uber",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white",
        )
        title_label.pack(pady=20)

        # Frame principal avec scrollbar
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Mode de saisie
        mode_frame = tk.LabelFrame(
            main_frame, text="Mode de saisie", font=("Arial", 12, "bold"), bg="#f0f0f0"
        )
        mode_frame.pack(fill=tk.X, pady=10)

        tk.Radiobutton(
            mode_frame,
            text="Coordonnées GPS (calcul automatique de la distance)",
            variable=self.input_mode,
            value="coordinates",
            command=self.toggle_input_mode,
            font=("Arial", 10),
            bg="#f0f0f0",
        ).pack(anchor=tk.W, padx=10, pady=5)

        tk.Radiobutton(
            mode_frame,
            text="Distance directement",
            variable=self.input_mode,
            value="distance",
            command=self.toggle_input_mode,
            font=("Arial", 10),
            bg="#f0f0f0",
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Frame pour les coordonnées GPS
        self.coords_frame = tk.LabelFrame(
            main_frame,
            text="Coordonnées GPS",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
        )
        self.coords_frame.pack(fill=tk.X, pady=10)

        self.create_coordinate_inputs()

        # Frame pour la distance directe
        self.distance_frame = tk.LabelFrame(
            main_frame, text="Distance", font=("Arial", 12, "bold"), bg="#f0f0f0"
        )
        self.distance_frame.pack(fill=tk.X, pady=10)

        self.create_distance_input()

        # Frame pour les autres informations
        info_frame = tk.LabelFrame(
            main_frame,
            text="Informations du trajet",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
        )
        info_frame.pack(fill=tk.X, pady=10)

        self.create_info_inputs(info_frame)

        # Boutons
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, pady=20)

        predict_btn = tk.Button(
            button_frame,
            text="🔮 Prédire le Tarif",
            font=("Arial", 14, "bold"),
            bg="#27ae60",
            fg="white",
            command=self.predict_fare,
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10,
        )
        predict_btn.pack(side=tk.LEFT, padx=10)

        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Effacer",
            font=("Arial", 12),
            bg="#e74c3c",
            fg="white",
            command=self.clear_inputs,
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=8,
        )
        clear_btn.pack(side=tk.LEFT, padx=10)

        # Affichage du résultat
        self.result_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
        )
        self.result_label.pack(pady=10)

        # Initialisation du mode
        self.toggle_input_mode()

    def create_coordinate_inputs(self):
        """Crée les champs de saisie pour les coordonnées GPS."""
        # Pickup
        tk.Label(
            self.coords_frame,
            text="📍 Point de départ (Pickup)",
            font=("Arial", 11, "bold"),
            bg="#f0f0f0",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        pickup_frame = tk.Frame(self.coords_frame, bg="#f0f0f0")
        pickup_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(pickup_frame, text="Latitude:", font=("Arial", 10), bg="#f0f0f0").pack(
            side=tk.LEFT, padx=5
        )
        self.entry_pickup_lat = tk.Entry(pickup_frame, font=("Arial", 11), width=15)
        self.entry_pickup_lat.pack(side=tk.LEFT, padx=5)

        tk.Label(pickup_frame, text="Longitude:", font=("Arial", 10), bg="#f0f0f0").pack(
            side=tk.LEFT, padx=5
        )
        self.entry_pickup_lon = tk.Entry(pickup_frame, font=("Arial", 11), width=15)
        self.entry_pickup_lon.pack(side=tk.LEFT, padx=5)

        # Dropoff
        tk.Label(
            self.coords_frame,
            text="🎯 Point d'arrivée (Dropoff)",
            font=("Arial", 11, "bold"),
            bg="#f0f0f0",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        dropoff_frame = tk.Frame(self.coords_frame, bg="#f0f0f0")
        dropoff_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(dropoff_frame, text="Latitude:", font=("Arial", 10), bg="#f0f0f0").pack(
            side=tk.LEFT, padx=5
        )
        self.entry_dropoff_lat = tk.Entry(dropoff_frame, font=("Arial", 11), width=15)
        self.entry_dropoff_lat.pack(side=tk.LEFT, padx=5)

        tk.Label(dropoff_frame, text="Longitude:", font=("Arial", 10), bg="#f0f0f0").pack(
            side=tk.LEFT, padx=5
        )
        self.entry_dropoff_lon = tk.Entry(dropoff_frame, font=("Arial", 11), width=15)
        self.entry_dropoff_lon.pack(side=tk.LEFT, padx=5)

        # Distance calculée (affichage seulement)
        self.distance_calculated_label = tk.Label(
            self.coords_frame,
            text="Distance calculée : -- km",
            font=("Arial", 10, "italic"),
            bg="#f0f0f0",
            fg="#7f8c8d",
        )
        self.distance_calculated_label.pack(pady=5)

    def create_distance_input(self):
        """Crée le champ de saisie pour la distance directe."""
        distance_input_frame = tk.Frame(self.distance_frame, bg="#f0f0f0")
        distance_input_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            distance_input_frame,
            text="Distance (km):",
            font=("Arial", 11),
            bg="#f0f0f0",
        ).pack(side=tk.LEFT, padx=5)

        self.entry_distance_direct = tk.Entry(distance_input_frame, font=("Arial", 11), width=20)
        self.entry_distance_direct.pack(side=tk.LEFT, padx=5)

    def create_info_inputs(self, parent_frame):
        """Crée les champs pour les autres informations."""
        # Nombre de passagers
        passenger_frame = tk.Frame(parent_frame, bg="#f0f0f0")
        passenger_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(passenger_frame, text="👥 Nombre de passagers:", font=("Arial", 10), bg="#f0f0f0").pack(
            side=tk.LEFT, padx=5
        )
        self.entry_passenger_count = tk.Entry(passenger_frame, font=("Arial", 11), width=10)
        self.entry_passenger_count.pack(side=tk.LEFT, padx=5)
        self.entry_passenger_count.insert(0, "1")

        # Date et heure
        datetime_frame = tk.Frame(parent_frame, bg="#f0f0f0")
        datetime_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(datetime_frame, text="📅 Date et heure:", font=("Arial", 10), bg="#f0f0f0").pack(
            side=tk.LEFT, padx=5
        )

        # Utiliser la date/heure actuelle par défaut
        now = datetime.now()
        self.entry_pickup_year = tk.Entry(datetime_frame, font=("Arial", 10), width=8)
        self.entry_pickup_year.insert(0, str(now.year))
        self.entry_pickup_year.pack(side=tk.LEFT, padx=2)

        self.entry_pickup_month = tk.Entry(datetime_frame, font=("Arial", 10), width=5)
        self.entry_pickup_month.insert(0, str(now.month))
        self.entry_pickup_month.pack(side=tk.LEFT, padx=2)

        self.entry_pickup_day = tk.Entry(datetime_frame, font=("Arial", 10), width=5)
        self.entry_pickup_day.insert(0, str(now.day))
        self.entry_pickup_day.pack(side=tk.LEFT, padx=2)

        self.entry_pickup_hour = tk.Entry(datetime_frame, font=("Arial", 10), width=5)
        self.entry_pickup_hour.insert(0, str(now.hour))
        self.entry_pickup_hour.pack(side=tk.LEFT, padx=2)

        tk.Label(datetime_frame, text="(Année Mois Jour Heure)", font=("Arial", 9, "italic"), bg="#f0f0f0", fg="#7f8c8d").pack(
            side=tk.LEFT, padx=5
        )

    def toggle_input_mode(self):
        """Affiche/masque les frames selon le mode sélectionné."""
        if self.input_mode.get() == "coordinates":
            self.coords_frame.pack(fill=tk.X, pady=10)
            self.distance_frame.pack_forget()
        else:
            self.coords_frame.pack_forget()
            self.distance_frame.pack(fill=tk.X, pady=10)

    def calculate_distance_from_coords(self):
        """Calcule la distance à partir des coordonnées GPS."""
        try:
            pickup_lat = float(self.entry_pickup_lat.get())
            pickup_lon = float(self.entry_pickup_lon.get())
            dropoff_lat = float(self.entry_dropoff_lat.get())
            dropoff_lon = float(self.entry_dropoff_lon.get())

            distance = haversine(pickup_lon, dropoff_lon, pickup_lat, dropoff_lat)
            self.distance_calculated_label.config(
                text=f"Distance calculée : {distance:.2f} km", fg="#27ae60"
            )
            return distance
        except ValueError:
            self.distance_calculated_label.config(
                text="Distance calculée : Erreur - Vérifiez les coordonnées", fg="#e74c3c"
            )
            return None

    def validate_inputs(self):
        """Valide tous les champs de saisie."""
        errors = []

        # Validation selon le mode
        if self.input_mode.get() == "coordinates":
            try:
                pickup_lat = float(self.entry_pickup_lat.get())
                pickup_lon = float(self.entry_pickup_lon.get())
                dropoff_lat = float(self.entry_dropoff_lat.get())
                dropoff_lon = float(self.entry_dropoff_lon.get())

                if not (-90 <= pickup_lat <= 90) or not (-90 <= dropoff_lat <= 90):
                    errors.append("Les latitudes doivent être entre -90 et 90")
                if not (-180 <= pickup_lon <= 180) or not (-180 <= dropoff_lon <= 180):
                    errors.append("Les longitudes doivent être entre -180 et 180")

                distance = self.calculate_distance_from_coords()
                if distance is None or distance <= 0:
                    errors.append("Distance invalide calculée à partir des coordonnées")
                elif distance > 60:
                    errors.append(f"Distance trop grande ({distance:.2f} km). Maximum: 60 km")

            except ValueError:
                errors.append("Coordonnées GPS invalides (doivent être des nombres)")
        else:
            try:
                distance = float(self.entry_distance_direct.get())
                if distance <= 0:
                    errors.append("La distance doit être positive")
                elif distance > 60:
                    errors.append(f"Distance trop grande ({distance:.2f} km). Maximum: 60 km")
            except ValueError:
                errors.append("Distance invalide (doit être un nombre)")

        # Validation du nombre de passagers
        try:
            passenger_count = int(self.entry_passenger_count.get())
            if passenger_count < 1 or passenger_count >= 10:
                errors.append("Le nombre de passagers doit être entre 1 et 9")
        except ValueError:
            errors.append("Nombre de passagers invalide")

        # Validation de la date/heure
        try:
            year = int(self.entry_pickup_year.get())
            month = int(self.entry_pickup_month.get())
            day = int(self.entry_pickup_day.get())
            hour = int(self.entry_pickup_hour.get())

            if not (2000 <= year <= 2030):
                errors.append("L'année doit être entre 2000 et 2030")
            if not (1 <= month <= 12):
                errors.append("Le mois doit être entre 1 et 12")
            if not (1 <= day <= 31):
                errors.append("Le jour doit être entre 1 et 31")
            if not (0 <= hour <= 23):
                errors.append("L'heure doit être entre 0 et 23")

            # Calculer le jour de la semaine
            try:
                date_obj = datetime(year, month, day)
                dayofweek = date_obj.weekday()  # 0 = lundi, 6 = dimanche
            except ValueError:
                errors.append("Date invalide")

        except ValueError:
            errors.append("Date/heure invalides (doivent être des nombres entiers)")

        return errors, dayofweek if "dayofweek" in locals() else None

    def predict_fare(self):
        """Effectue la prédiction du tarif."""
        if self.model is None:
            messagebox.showerror(
                "Erreur",
                "Le modèle n'a pas pu être chargé. Veuillez exécuter main.py d'abord.",
            )
            return

        # Validation
        errors, dayofweek = self.validate_inputs()
        if errors:
            messagebox.showerror("Erreurs de validation", "\n".join(errors))
            return

        try:
            # Récupération des valeurs
            passenger_count = int(self.entry_passenger_count.get())
            year = int(self.entry_pickup_year.get())
            month = int(self.entry_pickup_month.get())
            day = int(self.entry_pickup_day.get())
            hour = int(self.entry_pickup_hour.get())

            # Distance
            if self.input_mode.get() == "coordinates":
                distance = self.calculate_distance_from_coords()
                pickup_lat = float(self.entry_pickup_lat.get())
                dropoff_lat = float(self.entry_dropoff_lat.get())
            else:
                distance = float(self.entry_distance_direct.get())
                # Pour le modèle, on a besoin des latitudes même si on n'utilise que la distance
                # On peut mettre des valeurs par défaut (ex: coordonnées de NYC)
                pickup_lat = 40.7128  # Latitude par défaut (NYC)
                dropoff_lat = 40.7128

            # Création du DataFrame d'entrée (même ordre que dans le preprocessing)
            input_data = pd.DataFrame(
                {
                    "pickup_latitude": [pickup_lat],
                    "dropoff_latitude": [dropoff_lat],
                    "passenger_count": [passenger_count],
                    "pickup_year": [year],
                    "pickup_month": [month],
                    "pickup_day": [day],
                    "pickup_hour": [hour],
                    "pickup_dayofweek": [dayofweek],
                    "Distance": [distance],
                }
            )

            # Le RandomForest est entraîné sur des données non scalées
            # donc on fait la prédiction directement sans scaler
            fare_prediction = self.model.predict(input_data)[0]

            # Affichage du résultat
            self.result_label.config(
                text=f"💰 Tarif prédit : ${fare_prediction:.2f}",
                fg="#27ae60",
            )

            messagebox.showinfo(
                "Prédiction réussie",
                f"Le tarif estimé pour ce trajet est de :\n\n${fare_prediction:.2f}",
            )

        except Exception as e:
            messagebox.showerror("Erreur de prédiction", f"Une erreur est survenue : {str(e)}")
            self.result_label.config(text="❌ Erreur lors de la prédiction", fg="#e74c3c")

    def clear_inputs(self):
        """Efface tous les champs de saisie."""
        # Coordonnées
        self.entry_pickup_lat.delete(0, tk.END)
        self.entry_pickup_lon.delete(0, tk.END)
        self.entry_dropoff_lat.delete(0, tk.END)
        self.entry_dropoff_lon.delete(0, tk.END)
        self.distance_calculated_label.config(text="Distance calculée : -- km", fg="#7f8c8d")

        # Distance directe
        self.entry_distance_direct.delete(0, tk.END)

        # Autres champs
        self.entry_passenger_count.delete(0, tk.END)
        self.entry_passenger_count.insert(0, "1")

        # Date/heure actuelle
        now = datetime.now()
        self.entry_pickup_year.delete(0, tk.END)
        self.entry_pickup_year.insert(0, str(now.year))
        self.entry_pickup_month.delete(0, tk.END)
        self.entry_pickup_month.insert(0, str(now.month))
        self.entry_pickup_day.delete(0, tk.END)
        self.entry_pickup_day.insert(0, str(now.day))
        self.entry_pickup_hour.delete(0, tk.END)
        self.entry_pickup_hour.insert(0, str(now.hour))

        # Résultat
        self.result_label.config(text="")


def main():
    """Fonction principale pour lancer l'application."""
    root = tk.Tk()
    app = UberFarePredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

