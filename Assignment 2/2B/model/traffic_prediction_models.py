import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import time
import sys
sys.path.append('dataset')
from data import process_data

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_lstm_model(input_shape, neurons=50, dropout_rate=0.2):
    """Build an LSTM model."""
    model = Sequential()
    model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons//2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

def build_gru_model(input_shape, neurons=50, dropout_rate=0.2):
    """Build a GRU model."""
    model = Sequential()
    model.add(GRU(neurons, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(GRU(neurons//2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

def build_bidirectional_lstm_model(input_shape, neurons=50, dropout_rate=0.2):
    """Build a Bidirectional LSTM model."""
    model = Sequential()
    model.add(Bidirectional(LSTM(neurons, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(neurons//2)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name, epochs=50, batch_size=32):
    """Train and evaluate a model."""
    print(f"\n{'-'*20} Training {model_name} {'-'*20}")
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        f'{model_name}_best_model.keras',  # Changed from .h5 to .keras
        save_best_only=True
    )
    
    # Training time measurement
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Prediction time measurement
    start_time = time.time()
    y_pred = model.predict(x_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print(f"\n{model_name} Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    
    return history, y_pred, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'prediction_time': prediction_time
    }

def plot_results(y_test, predictions_dict, scaler, title="Traffic Flow Prediction Comparison"):
    """Plot the prediction results for all models."""
    plt.figure(figsize=(15, 8))
    
    # If data was scaled, we need to invert the transformation for better visualization
    if scaler:
        # Original inverse transform
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Inverse transform predictions
        for model_name, pred in predictions_dict.items():
            predictions_dict[model_name] = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    else:
        y_test_orig = y_test
    
    # Plot ground truth
    plt.plot(y_test_orig, label='Actual', linewidth=2)
    
    # Plot predictions for each model
    for model_name, pred in predictions_dict.items():
        plt.plot(pred, label=f'{model_name} Prediction', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Flow')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('traffic_predictions_comparison.png')
    plt.show()

def plot_learning_curves(histories_dict, title="Learning Curves Comparison"):
    """Plot the learning curves for all models."""
    plt.figure(figsize=(15, 8))
    
    for model_name, history in histories_dict.items():
        plt.plot(history.history['loss'], label=f'{model_name} Training Loss')
        plt.plot(history.history['val_loss'], label=f'{model_name} Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curves_comparison.png')
    plt.show()

def compare_models(metrics_dict):
    """Compare the performance metrics of all models."""
    # Create a DataFrame for comparison
    metrics_df = pd.DataFrame().from_dict(metrics_dict, orient='index')
    
    # Print the comparison table
    print("\n" + "="*50)
    print("Model Performance Comparison")
    print("="*50)
    print(metrics_df)
    
    # Find the best model for each metric
    best_model = {}
    for metric in ['mse', 'rmse', 'mae']:
        best_model[metric] = metrics_df[metric].idxmin()
    
    best_model['r2'] = metrics_df['r2'].idxmax()
    best_model['fastest_training'] = metrics_df['training_time'].idxmin()
    best_model['fastest_prediction'] = metrics_df['prediction_time'].idxmin()
    
    print("\n" + "="*50)
    print("Best Models by Metric")
    print("="*50)
    for metric, model in best_model.items():
        print(f"Best model for {metric}: {model}")
    
    return metrics_df, best_model

def main():
    """Main function to run the TBRGS system."""
    # Configuration parameters
    file_path = 'dataset/Scats_Data_October_2006.csv'
    lags = 12  # Number of time steps to use as features
    train_ratio = 0.75
    epochs = 50
    batch_size = 32
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    x_train, y_train, x_test, y_test, scaler, time_train, time_test =process_data(
        file_path, lags, train_ratio
    )
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")
    
    # Model input shape
    input_shape = (x_train.shape[1], x_train.shape[2])
    
    # Build models
    models = {
        'LSTM': build_lstm_model(input_shape),
        'GRU': build_gru_model(input_shape),
        'Bidirectional_LSTM': build_bidirectional_lstm_model(input_shape)
    }
    
    # Train and evaluate models
    histories = {}
    predictions = {}
    metrics = {}
    
    for model_name, model in models.items():
        history, prediction, model_metrics = train_and_evaluate_model(
            model, x_train, y_train, x_test, y_test, model_name, epochs, batch_size
        )
        histories[model_name] = history
        predictions[model_name] = prediction
        metrics[model_name] = model_metrics
    
    # Plot results
    plot_results(y_test, predictions, scaler)
    plot_learning_curves(histories)
    
    # Compare models
    metrics_df, best_model = compare_models(metrics)
    
    # Save model comparison to CSV
    metrics_df.to_csv('model_comparison.csv')
    
    print("\nTraining and evaluation completed!")
    print("Results have been saved to 'model_comparison.csv'.")
    print("Plots have been saved as 'traffic_predictions_comparison.png' and 'learning_curves_comparison.png'.")

if __name__ == "__main__":
    main()
