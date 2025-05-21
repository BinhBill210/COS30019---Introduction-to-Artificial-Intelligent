"""
Data processing module for SCATS traffic data analysis.
This module provides functions for processing and preparing SCATS traffic data for analysis.
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Union, Optional


def get_coords(data: str, scats: str, junction: str) -> Tuple[float, float]:
    """
    Calculate the coordinates for a specific SCATS junction.
    
    Args:
        data (str): Path to the SCATS data CSV file
        scats (str): SCATS number to filter by
        junction (str): Junction direction (N, S, E, W, NE, NW, SE, SW)
    
    Returns:
        Tuple[float, float]: Latitude and longitude of the junction, or (-1, -1) if not found
    """
    # Read and filter data
    df = pd.read_csv(data, encoding='utf-8').fillna(0)
    filtered_df = df[df['SCATS Number'] == int(scats)].drop_duplicates(
        subset=['NB_LATITUDE', 'NB_LONGITUDE']
    ).reset_index()
    
    if len(filtered_df) == 0:
        return -1, -1
    
    # Calculate center point
    lat = filtered_df['NB_LATITUDE'].mean()
    long = filtered_df['NB_LONGITUDE'].mean()
    
    # Find the closest point matching the junction direction
    for idx, row in filtered_df.iterrows():
        lat_diff = float(row['NB_LATITUDE']) - lat
        long_diff = float(row['NB_LONGITUDE']) - long
        
        # Calculate angle and determine direction
        if abs(lat_diff) > abs(long_diff):
            angle = math.degrees(math.atan(long_diff/lat_diff))
            if lat_diff > 0:
                if angle > 22:
                    if long_diff > 0 and junction == "NE":
                        return row['NB_LATITUDE'], row['NB_LONGITUDE']
                    elif long_diff < 0 and junction == "NW":
                        return row['NB_LATITUDE'], row['NB_LONGITUDE']
                elif junction == "N":
                    return row['NB_LATITUDE'], row['NB_LONGITUDE']
            else:
                if angle > 22:
                    if long_diff > 0 and junction == "SE":
                        return row['NB_LATITUDE'], row['NB_LONGITUDE']
                    elif long_diff < 0 and junction == "SW":
                        return row['NB_LATITUDE'], row['NB_LONGITUDE']
                elif junction == "S":
                    return row['NB_LATITUDE'], row['NB_LONGITUDE']
        else:
            angle = math.degrees(math.atan(lat_diff/long_diff))
            if long_diff > 0:
                if angle > 22:
                    if lat_diff > 0 and junction == "NE":
                        return row['NB_LATITUDE'], row['NB_LONGITUDE']
                    elif lat_diff < 0 and junction == "SE":
                        return row['NB_LATITUDE'], row['NB_LONGITUDE']
                elif junction == "E":
                    return row['NB_LATITUDE'], row['NB_LONGITUDE']
            else:
                if angle > 22:
                    if lat_diff > 0 and junction == "NW":
                        return row['NB_LATITUDE'], row['NB_LONGITUDE']
                    elif lat_diff < 0 and junction == "SW":
                        return row['NB_LATITUDE'], row['NB_LONGITUDE']
                elif junction == "W":
                    return row['NB_LATITUDE'], row['NB_LONGITUDE']
    
    return -1, -1


def process_data(
    data_path: str,
    lags: int,
    train_ratio: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, np.ndarray, np.ndarray]:
    """
    Process time series data from CSV file, split into train/test sets, normalize and create series with delay.
    
    Args:
        data_path (str): Path to the SCATS data CSV file
        lags (int): Number of time steps to use as input features
        train_ratio (float): Ratio of data to use for training (default: 0.75)
    
    Returns:
        Tuple containing:
        - x_train: Training input features
        - y_train: Training target values
        - x_test: Test input features
        - y_test: Test target values
        - scaler: Fitted MinMaxScaler for inverse transformation
        - time_train: Training timestamps
        - time_test: Test timestamps
    """
    # Read and preprocess data
    df = pd.read_csv(data_path, encoding='utf-8', header=1).fillna(0)
    
    # Extract flow data and reshape for normalization
    flow_data = df.loc[:, 'V00':'V95'].values
    flow_data_reshaped = flow_data.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    flow_normalized = scaler.fit_transform(flow_data_reshaped)
    
    # Get timestamps
    time_col = df['Date'].values if 'Date' in df.columns else np.arange(len(flow_normalized))
    
    # Create time series with delay
    X, y, time_arr = [], [], []
    n = min(len(flow_normalized), len(time_col))
    
    for i in range(lags, n):
        X.append(flow_normalized[i - lags:i])
        y.append(flow_normalized[i])
        time_arr.append(time_col[i])
    
    # Convert to numpy arrays and reshape for LSTM input
    X = np.array(X)
    y = np.array(y)
    time_arr = np.array(time_arr)
    
    # Reshape X to (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split into train/test sets
    train_size = int(len(X) * train_ratio)
    x_train, y_train = X[:train_size], y[:train_size]
    x_test, y_test = X[train_size:], y[train_size:]
    time_train, time_test = time_arr[:train_size], time_arr[train_size:]
    
    return x_train, y_train, x_test, y_test, scaler, time_train, time_test


if __name__ == "__main__":
    # Example usage
    data_path = "2B/dataset/Scats_Data_October_2006.csv"
    x_train, y_train, x_test, y_test, scaler, time_train, time_test = process_data(
        data_path, lags=12, train_ratio=0.75
    )
    
    # Print shapes and samples
    print("Data Shapes:")
    print(f"X_train: {x_train.shape}")
    print(f"Y_train: {y_train.shape}")
    print(f"X_test: {x_test.shape}")
    print(f"Y_test: {y_test.shape}")
    print(f"Time_train: {time_train.shape}")
    print(f"Time_test: {time_test.shape}")
    
    print("\nSample Data:")
    print(f"X_train sample:\n{x_train[:2]}")
    print(f"Y_train sample:\n{y_train[:2]}")
    print(f"Time_train sample:\n{time_train[:2]}")
    print(f"Time_test sample:\n{time_test[:2]}")
    print(f"X_test sample:\n{x_test[:2]}")
    print(f"Y_test sample:\n{y_test[:2]}")