import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import joblib

pipeline = joblib.load('random_forest_model.pkl')  # Le modèle doit être dans le même dossier que app.py

def predict_fare():
    try:
        pickup_latitude = float(entry_pickup_latitude.get())
        dropoff_latitude = float(entry_dropoff_latitude.get())
        passenger_count = int(entry_passenger_count.get())
        pickup_year = int(entry_pickup_year.get())
        pickup_month = int(entry_pickup_month.get())
        pickup_day = int(entry_pickup_day.get())
        pickup_hour = int(entry_pickup_hour.get())
        pickup_dayofweek = int(entry_pickup_dayofweek.get())
        distance = float(entry_distance.get())

        if passenger_count < 1 or passenger_count > 6:
            raise ValueError("Passenger count must be between 1 and 6.")

        if pickup_year < 2025 or pickup_year > 2030:
            raise ValueError("Pickup year must be between 2025 and 2030.")

        if pickup_month < 1 or pickup_month > 12:
            raise ValueError("Pickup month must be between 1 and 12.")

        if pickup_day < 1 or pickup_day > 30:
            raise ValueError("Pickup day must be between 1 and 30.")

        if pickup_hour < 1 or pickup_hour > 24:
            raise ValueError("Pickup hour must be between 1 and 24.")

        if pickup_dayofweek < 1 or pickup_dayofweek > 7:
            raise ValueError("Pickup day of week must be between 1 and 7.")

        if distance < 1:
            raise ValueError("Distance must be at least 1 km.")

        input_data = pd.DataFrame({
            'pickup_latitude': [pickup_latitude],
            'dropoff_latitude': [dropoff_latitude],
            'passenger_count': [passenger_count],
            'pickup_year': [pickup_year],
            'pickup_month': [pickup_month],
            'pickup_day': [pickup_day],
            'pickup_hour': [pickup_hour],
            'pickup_dayofweek': [pickup_dayofweek],
            'Distance': [distance]
        })

        fare_amount = pipeline.predict(input_data)[0]

        messagebox.showinfo("Prediction", f"Predicted Fare Amount: ${fare_amount:.2f}")

        response = messagebox.askquestion("More Predictions", "Do you want to make more predictions?")
        if response == "yes":
            clear_inputs()
        else:
            window.destroy()

    except ValueError as ve:
        messagebox.showerror("Invalid Input", str(ve))

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def clear_inputs():
    entry_pickup_latitude.delete(0, tk.END)
    entry_dropoff_latitude.delete(0, tk.END)
    entry_passenger_count.delete(0, tk.END)
    entry_pickup_year.delete(0, tk.END)
    entry_pickup_month.delete(0, tk.END)
    entry_pickup_day.delete(0, tk.END)
    entry_pickup_hour.delete(0, tk.END)
    entry_pickup_dayofweek.delete(0, tk.END)
    entry_distance.delete(0, tk.END)

window = tk.Tk()
window.title("Uber Fare Prediction")
window.geometry("600x500")
window.configure(cursor="arrow")

bg_image = Image.open("../uber.jpg")
bg_image = bg_image.resize((600, 500), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(window, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)
bg_label.image = bg_photo

title_label = tk.Label(window, text="Uber Fare Prediction", font=("Arial", 24, "bold"), bg="white", fg="black")
title_label.place(relx=0.5, rely=0.05, anchor="n")

def create_label_and_entry(label_text, row):
    tk.Label(window, text=label_text, font=("Arial", 12), bg="#f0f0f0").place(
        relx=0.1, rely=0.1 + row * 0.07, anchor="w"
    )
    entry = tk.Entry(window, font=("Arial", 12))
    entry.place(relx=0.5, rely=0.1 + row * 0.07, anchor="w")
    return entry

entry_pickup_latitude = create_label_and_entry("Pickup Latitude:", 1)
entry_dropoff_latitude = create_label_and_entry("Dropoff Latitude:", 2)
entry_passenger_count = create_label_and_entry("Passenger Count:", 3)
entry_pickup_year = create_label_and_entry("Pickup Year:", 4)
entry_pickup_month = create_label_and_entry("Pickup Month:", 5)
entry_pickup_day = create_label_and_entry("Pickup Day:", 6)
entry_pickup_hour = create_label_and_entry("Pickup Hour:", 7)
entry_pickup_dayofweek = create_label_and_entry("Pickup Day of Week:", 8)
entry_distance = create_label_and_entry("Distance (km):", 9)

predict_button = tk.Button(window, text="Predict", font=("Arial", 14, "bold"), bg="white", fg="black", command=predict_fare)
predict_button.place(relx=0.5, rely=0.9, anchor="s")

window.mainloop()
