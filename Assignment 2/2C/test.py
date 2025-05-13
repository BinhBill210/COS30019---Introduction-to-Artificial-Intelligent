"""
Processing the data
"""
#import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression



def get_coords(data, scats, junction):
    # Read data file
    df = pd.read_csv('dataset/Scats_Data_October_2006.csv', encoding='utf-8').fillna(0)

    # Filter the DataFrame around the SCATS, making a new DataFrame that just has this SCAT
    filtered_df = df[df['SCATS Number'] == int(scats)]

    # Remove duplicates if there are any
    filtered_df = filtered_df.drop_duplicates(subset=['NB_LATITUDE', 'NB_LONGITUDE'])
    filtered_df = filtered_df.reset_index()
    i = 0
    lat = 0
    long = 0
    #Calculate the aprx centre of all the junctions. This has flaws and isn't perfect for say, a 90 degree turn
    while ((i + 1) < len(filtered_df)):
        lat = lat + float(filtered_df.loc[i,'NB_LATITUDE'])
        long = long + float(filtered_df.loc[i,'NB_LONGITUDE'])
        i += 1
    lat = lat/i
    long = long/i
    safeIndex = -1
    i = 0
    #Calculate the direction the exit to a junction is
    while ((i + 1) < len(filtered_df)):
        tempa = float(filtered_df.loc[i,'NB_LATITUDE']) - lat
        tempo = float(filtered_df.loc[i,'NB_LONGITUDE']) - long
        if ( abs(tempa) > abs(tempo)):
            angle = math.degrees(math.atan(tempo/tempa))
            if (tempa > 0):
                if (angle > 22):
                    if (tempo > 0):
                        if (junction == "NE"):
                            safeIndex = i
                    else:
                        if (junction == "NW"):
                            safeIndex = i
                else:
                    if (junction == "N"):
                        safeIndex = i
            else:
                if (angle > 22):
                    if (tempo > 0):
                        if (junction == "SE"):
                            safeIndex = i
                    else:
                        if (junction == "SW"):
                            safeIndex = i
                else:
                    if (junction == "S"):
                        safeIndex = i
        else:
            angle = math.degrees(math.atan(tempa/tempo))
            if (tempo > 0):
                if (angle > 22):
                    if (tempa > 0):
                        if (junction == "NE"):
                            safeIndex = i
                    else:
                        if (junction == "SE"):
                            safeIndex = i
                else:
                    if (junction == "E"):
                        safeIndex = i
            else:
                if (angle > 22):
                    if (tempa > 0):
                        if (junction == "NW"):
                            safeIndex = i
                    else:
                        if (junction == "SW"):
                            safeIndex = i
                else:
                    if (junction == "E"):
                        safeIndex = i
        i += 1

    if (safeIndex != -1):
        return filtered_df.loc[safeIndex,'NB_LATITUDE'], filtered_df.loc[safeIndex,'NB_LONGITUDE']
    else:
        return -1, -1

def process_data(data, lags, train_ratio=0.75):
    """
    Process time series data from CSV file, split into train/test sets in time order, normalize and create series with delay.
    Returns: x_train, y_train, x_test, y_test, scaler, time_train, time_test
    """
    # Read in CSV file
    df = pd.read_csv('dataset/Scats_Data_October_2006.csv', encoding='utf-8',header=1).fillna(0) 
    #Reshapes the DataFrames to only take values from V00 to V95
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df.loc[:, 'V00':'V95'].values.reshape(-1, 1))
    flow = scaler.transform(df.loc[:, 'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]

    # Get the time column
    if 'Date' in df.columns:
        time_col = df['Date'].values
    else:
        time_col = np.arange(len(flow)) 

    # Create time series with delay
    X, y, time_arr = [], [], []
    n = min(len(flow), len(time_col))
    for i in range(lags, n):
        X.append(flow[i - lags: i])
        y.append(flow[i])
        time_arr.append(time_col[i])
    X = np.array(X)
    y = np.array(y)
    time_arr = np.array(time_arr)

    # Split train/test in the ratio of 75% first for train, 25% last for test
    train_size = int(len(X) * train_ratio)
    x_train, y_train = X[:train_size], y[:train_size]
    x_test, y_test = X[train_size:], y[train_size:]
    time_train, time_test = time_arr[:train_size], time_arr[train_size:]

    return x_train, y_train, x_test, y_test, scaler, time_train, time_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scaler, time_train, time_test = process_data('Scats_Data_October_2006.csv', lags=12, train_ratio=0.75)
    print("X_train shape:", x_train.shape)
    print("Y_train shape:", y_train.shape)
    print("X_test shape:", x_test.shape)
    print("Y_test shape:", y_test.shape)
    print("Time_train shape:", time_train.shape)
    print("Time_test shape:", time_test.shape)
    print("\nSample X_train:", x_train[:2])
    print("Sample Y_train:", y_train[:2])
    print("Sample time_train:", time_train[:2])
    print("Sample time_test:", time_test[:2])