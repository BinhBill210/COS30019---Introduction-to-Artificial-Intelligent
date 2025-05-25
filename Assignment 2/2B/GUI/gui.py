import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib # For loading the scaler
import random # For simulating current traffic data
import sys
# Assuming 'data.py' is in the same directory or accessible via PYTHONPATH
sys.path.append('dataset')
from data import process_data

class TBRGS_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Traffic-Based Route Guidance System (TBRGS)")
        master.geometry("600x450") # Increased size to accommodate more info
        master.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(master, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="TBRGS - Route Planner", font=("Inter", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # --- Model Loading ---
        self.lags = 12 # Must match the lags used during training
        self.load_models_and_scaler()

        # Current Location Input
        current_location_label = ttk.Label(main_frame, text="Current Location:", font=("Inter", 10))
        current_location_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        self.current_location_entry = ttk.Entry(main_frame, width=50)
        self.current_location_entry.grid(row=1, column=1, pady=5, padx=5)
        self.current_location_entry.insert(0, "e.g., 123 Main St, Boroondara")

        # Destination Input
        destination_label = ttk.Label(main_frame, text="Destination:", font=("Inter", 10))
        destination_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        self.destination_entry = ttk.Entry(main_frame, width=50)
        self.destination_entry.grid(row=2, column=1, pady=5, padx=5)
        self.destination_entry.insert(0, "e.g., 456 Elm St, Melbourne CBD")

        # Calculate Route Button
        calculate_button = ttk.Button(main_frame, text="Calculate Route", command=self.calculate_route)
        calculate_button.grid(row=3, column=0, columnspan=2, pady=20)

        # Estimated Time Display
        self.estimated_time_label = ttk.Label(main_frame, text="Estimated Time: --", font=("Inter", 12, "italic"))
        self.estimated_time_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Individual Model Predictions Display
        self.lstm_pred_label = ttk.Label(main_frame, text="LSTM Prediction: --", font=("Inter", 10))
        self.lstm_pred_label.grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        self.gru_pred_label = ttk.Label(main_frame, text="GRU Prediction: --", font=("Inter", 10))
        self.gru_pred_label.grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        self.bidirectional_lstm_pred_label = ttk.Label(main_frame, text="Bi-LSTM Prediction: --", font=("Inter", 10))
        self.bidirectional_lstm_pred_label.grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        # Configure grid column weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=3) # Give more space to entry fields

    def load_models_and_scaler(self):
        """Loads the trained ML models and the MinMaxScaler."""
        try:
            self.lstm_model = load_model('models/LSTM_best_model.h5')
            self.gru_model = load_model('models/GRU_best_model.h5')
            self.bidirectional_lstm_model = load_model('models/Bidirectional_LSTM_best_model.h5')
            self.scaler = joblib.load('models/scaler.pkl')
            print("Models and scaler loaded successfully.")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Could not load models or scaler: {e}\n"
                                                       "Please ensure training script was run and generated .h5 and .pkl files.")
            self.master.destroy() # Close the app if models can't be loaded

    def flow_to_time_heuristic(self, predicted_flow_value):
        """
        Converts a predicted traffic flow value (original scale) to an estimated time.
        This is a simplified heuristic. Higher flow means lower travel time.
        """
        # Ensure scaler was loaded to get min/max data for normalization
        if not hasattr(self, 'scaler') or self.scaler is None:
            return "N/A (Scaler not loaded)"

        # Get min/max flow from the scaler's fitted data
        min_flow = self.scaler.data_min_[0]
        max_flow = self.scaler.data_max_[0]
        flow_range = max_flow - min_flow

        # Define a reasonable min/max travel time for the heuristic (e.g., in minutes)
        # These values would ideally be calibrated based on route distance
        max_travel_time_min = 60 # Max possible travel time for a segment
        min_travel_time_min = 5  # Min possible travel time for a segment

        if flow_range == 0: # Avoid division by zero if all traffic data was constant
            return (max_travel_time_min + min_travel_time_min) / 2

        # Normalize the predicted flow value to a 0-1 range based on the training data's flow range
        normalized_predicted_flow = (predicted_flow_value - min_flow) / flow_range

        # Invert the normalized flow to get a time factor:
        # If flow is high (closer to 1), time factor is low (closer to 0)
        # If flow is low (closer to 0), time factor is high (closer to 1)
        time_factor = 1 - normalized_predicted_flow

        # Scale the time factor to the desired travel time range
        estimated_time = min_travel_time_min + time_factor * (max_travel_time_min - min_travel_time_min)

        # Clamp the estimated time within the defined bounds
        return max(min_travel_time_min, min(max_travel_time_min, estimated_time))


    def calculate_route(self):
        """
        Retrieves input, simulates traffic data, makes predictions, and displays results.
        """
        current_location = self.current_location_entry.get()
        destination = self.destination_entry.get()

        if not current_location or not destination:
            messagebox.showwarning("Input Error", "Please enter both current location and destination.")
            self.estimated_time_label.config(text="Estimated Time: --")
            self.lstm_pred_label.config(text="LSTM Prediction: --")
            self.gru_pred_label.config(text="GRU Prediction: --")
            self.bidirectional_lstm_pred_label.config(text="Bi-LSTM Prediction: --")
            return

        self.estimated_time_label.config(text="Calculating...")
        self.lstm_pred_label.config(text="LSTM Prediction: Calculating...")
        self.gru_pred_label.config(text="GRU Prediction: Calculating...")
        self.bidirectional_lstm_pred_label.config(text="Bi-LSTM Prediction: Calculating...")
        self.master.update_idletasks() # Update GUI to show "Calculating..."

        # --- Simulate Current Traffic Data ---
        # In a real system, you would query a map/traffic API for the route
        # and get the current traffic flow data for the relevant segments.
        # For this example, we'll load the full dataset and take a random slice
        # from the test set as our "current traffic snapshot".
        try:
            # We need to re-run process_data to get access to the full x_test data
            # This is not efficient for a live system, but fine for demonstration.
            _, _, x_test_full, _, _, _, _ = process_data('dataset/Scats_Data_October_2006.csv', self.lags, 0.75)

            if len(x_test_full) == 0:
                messagebox.showerror("Data Error", "No test data available to simulate current traffic. Check dataset.")
                return

            # Take a random sample from the test set as the "current" input
            random_idx = random.randint(0, len(x_test_full) - 1)
            current_traffic_input = x_test_full[random_idx:random_idx+1] # Shape (1, lags, 1)

            # Ensure the input is float32 for TensorFlow models
            current_traffic_input = current_traffic_input.astype(np.float32)

        except Exception as e:
            messagebox.showerror("Data Simulation Error", f"Could not simulate current traffic data: {e}")
            self.estimated_time_label.config(text="Estimated Time: Error")
            return

        # --- Make Predictions ---
        try:
            lstm_prediction_norm = self.lstm_model.predict(current_traffic_input)[0][0]
            gru_prediction_norm = self.gru_model.predict(current_traffic_input)[0][0]
            bidirectional_lstm_prediction_norm = self.bidirectional_lstm_model.predict(current_traffic_input)[0][0]

            # Inverse transform to get original traffic flow values
            lstm_prediction_orig = self.scaler.inverse_transform(np.array([[lstm_prediction_norm]]))[0][0]
            gru_prediction_orig = self.scaler.inverse_transform(np.array([[gru_prediction_norm]]))[0][0]
            bidirectional_lstm_prediction_orig = self.scaler.inverse_transform(np.array([[bidirectional_lstm_prediction_norm]]))[0][0]

            # --- Convert to Estimated Time (Heuristic) ---
            lstm_time = self.flow_to_time_heuristic(lstm_prediction_orig)
            gru_time = self.flow_to_time_heuristic(gru_prediction_orig)
            bidirectional_lstm_time = self.flow_to_time_heuristic(bidirectional_lstm_prediction_orig)

            # --- Display Results ---
            self.estimated_time_label.config(text="Predicted Travel Times:")
            self.lstm_pred_label.config(text=f"LSTM: {lstm_time:.1f} minutes (Flow: {lstm_prediction_orig:.2f})")
            self.gru_pred_label.config(text=f"GRU: {gru_time:.1f} minutes (Flow: {gru_prediction_orig:.2f})")
            self.bidirectional_lstm_pred_label.config(text=f"Bi-LSTM: {bidirectional_lstm_time:.1f} minutes (Flow: {bidirectional_lstm_prediction_orig:.2f})")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.estimated_time_label.config(text="Estimated Time: Error")
            self.lstm_pred_label.config(text="LSTM Prediction: Error")
            self.gru_pred_label.config(text="GRU Prediction: Error")
            self.bidirectional_lstm_pred_label.config(text="Bi-LSTM Prediction: Error")


if __name__ == "__main__":
    root = tk.Tk()
    app = TBRGS_GUI(root)
    root.mainloop()
