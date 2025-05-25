import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
import sys
sys.path.append('dataset')
from data import process_data

def load_best_model(model_name):
    """Load a saved model by name."""
    model_path = f'{model_name}_best_model.keras'
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Model {model_path} not found. Make sure to train models first.")

def estimate_travel_time(traffic_flow, distance, lanes=3, speed_limit=60, intersections=0):
    """
    Estimate travel time based on traffic flow using the quadratic model:
    flow = -1.4648375*(speed)² + 93.75*(speed)
    
    The relationship is divided into two regions:
    - Green line: When traffic is under capacity (uncongested)
    - Red line: When traffic is over capacity (congested)
    
    Args:
        traffic_flow: Predicted traffic flow in vehicles per hour
        distance: Distance in kilometers
        lanes: Number of lanes (default: 3)
        speed_limit: Speed limit in km/h (default: 60)
        intersections: Number of intersections on the route (default: 0)
        
    Returns:
        Estimated travel time in minutes
    """
    # Constants from the provided quadratic equation
    A = -1.4648375
    B = 93.75
    
    # The flow at capacity (turning point of the parabola)
    # This is where the green and red curves meet
    capacity_speed = 32  # km/h at capacity
    capacity_flow = 1500  # vehicles/hour at capacity
    
    # Flow at speed limit (351.56 vehicles/hour at 60 km/h)
    limit_flow = 351.56
    
    if traffic_flow <= 0:
        # For zero or negative flow, use speed limit
        speed = speed_limit
    elif traffic_flow <= limit_flow:
        # Flow is below the speed limit threshold
        # Use speed limit (blue dashed line)
        speed = speed_limit
    elif traffic_flow <= capacity_flow:
        # Flow is between speed limit threshold and capacity
        # Use the green curve (uncongested)
        # Solve the quadratic equation for speed
        discriminant = B**2 + 4*A*traffic_flow
        
        if discriminant < 0:
            # This shouldn't happen with our parameters
            speed = speed_limit * 0.8  # Fallback
        else:
            # Choose the larger solution (upper branch - green line)
            speed = (-B + np.sqrt(discriminant))/(2*A)
    else:
        # Flow is above capacity
        # Use the red curve (congested)
        discriminant = B**2 + 4*A*traffic_flow
        
        if discriminant < 0:
            # This shouldn't happen with our parameters
            speed = capacity_speed * 0.5  # Fallback for severe congestion
        else:
            # Choose the smaller solution (lower branch - red line)
            speed = (-B - np.sqrt(discriminant))/(2*A)
            
            # Ensure speed doesn't go below a minimum threshold
            speed = max(speed, 5)  # 5 km/h minimum speed
    
    # Calculate travel time in hours
    travel_time_hours = distance / speed
    
    # Convert to minutes
    travel_time_minutes = travel_time_hours * 60
    
    # Add 30 seconds (0.5 minutes) delay per intersection
    intersection_delay = intersections * 0.5  # 30 seconds = 0.5 minutes
    total_travel_time = travel_time_minutes + intersection_delay
    
    return total_travel_time

def predict_travel_times(model, x_data, distance, scaler=None, lanes=3, speed_limit=60):
    """
    Predict traffic flow and estimate travel times.
    
    Args:
        model: Trained traffic prediction model
        x_data: Input data for prediction
        distance: Distance in kilometers
        scaler: Scaler used for data normalization (optional)
        lanes: Number of lanes (default: 3)
        speed_limit: Speed limit in km/h (default: 60)
        
    Returns:
        Predicted traffic flows and estimated travel times
    """
    # Predict traffic flow
    y_pred = model.predict(x_data)
    
    # Inverse transform if scaler is provided
    if scaler:
        y_pred = scaler.inverse_transform(y_pred)
    
    # Estimate travel time for each predicted traffic flow
    travel_times = [estimate_travel_time(flow[0], distance, lanes, speed_limit) for flow in y_pred]
    
    return y_pred, travel_times

def plot_travel_times(times, predicted_times, model_name="Model"):
    """Plot actual vs predicted travel times."""
    plt.figure(figsize=(15, 6))
    plt.plot(times, label='Actual', linewidth=2)
    plt.plot(predicted_times, label=f'Predicted ({model_name})', alpha=0.7)
    plt.title(f'Actual vs Predicted Travel Times ({model_name})')
    plt.xlabel('Time Steps')
    plt.ylabel('Travel Time (minutes)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'travel_time_prediction_{model_name}.png')
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Estimate travel times using traffic prediction models.")
    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'GRU', 'Bidirectional_LSTM'],
                        help='Model to use for prediction (default: LSTM)')
    parser.add_argument('--distance', type=float, default=5.0,
                        help='Distance in kilometers (default: 5.0)')
    parser.add_argument('--lanes', type=int, default=3,
                        help='Number of lanes (default: 3)')
    parser.add_argument('--speed_limit', type=float, default=60.0,
                        help='Speed limit in km/h (default: 60.0)')
    parser.add_argument('--data_file', type=str, default='dataset/Scats_Data_October_2006.csv',
                        help='Path to traffic data CSV file')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    lags = 12
    train_ratio = 0.75
    x_train, y_train, x_test, y_test, scaler, time_train, time_test = process_data(
        args.data_file, lags, train_ratio
    )
    
    # Load the best model
    print(f"Loading {args.model} model...")
    model = load_best_model(args.model)
    
    # Predict traffic flow and estimate travel times for test data
    print(f"Estimating travel times for a {args.distance} km route with {args.lanes} lanes...")
    y_pred, travel_times = predict_travel_times(
        model, x_test, args.distance, scaler, args.lanes, args.speed_limit
    )
    
    # Calculate actual travel times from test data for comparison
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    actual_travel_times = [estimate_travel_time(flow[0], args.distance, args.lanes, args.speed_limit) 
                          for flow in y_test_scaled]
    
    # Calculate mean absolute error for travel time predictions
    mae = np.mean(np.abs(np.array(actual_travel_times) - np.array(travel_times)))
    print(f"Mean Absolute Error for Travel Time Prediction: {mae:.2f} minutes")
    
    # Plot the results
    plot_travel_times(actual_travel_times, travel_times, args.model)
    
    # Sample output for specific time periods
    print("\nSample Travel Time Predictions:")
    for i in range(0, len(travel_times), len(travel_times) // 5):
        print(f"Time Step {i}: Predicted: {travel_times[i]:.2f} min, Actual: {actual_travel_times[i]:.2f} min")
    
    print(f"\nTravel time prediction completed for {args.model}!")
    print(f"Results have been saved as 'travel_time_prediction_{args.model}.png'")

# Example usage in TrafficAwarePathFinder.update_edge_costs method:
def update_edge_costs(self, traffic_flows):
    """Update edge costs based on predicted traffic flows."""
    for edge in self.graph.edges():
        node1, node2 = edge
        distance = self.graph[node1][node2]['distance']
        
        # Get traffic flow for this edge
        flow = traffic_flows.get(edge, 0)
        
        # Each edge represents one intersection to pass through
        intersections = 1  
        
        # Estimate travel time based on traffic flow
        travel_time = estimate_travel_time(
            flow, distance, self.lanes, self.speed_limit, intersections
        )
        
        # Update edge cost
        self.graph[node1][node2]['cost'] = travel_time

if __name__ == "__main__":
    main()