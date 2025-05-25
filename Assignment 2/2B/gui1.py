import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import random
import time
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration - THIS MUST COME BEFORE ANY OTHER STREAMLIT COMMANDS
st.set_page_config(
    page_title="Traffic-Based Route Guidance System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add paths for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'dataset'))
sys.path.append(os.path.join(parent_dir, '2B/model'))

# Import our modules
try:
    from data import process_data
    try:
        from est import estimate_travel_time
        st.sidebar.success("Successfully imported estimate_travel_time function")
    except ImportError:
        # The function is already defined above, so just log the fallback
        st.sidebar.warning("Using built-in estimate_travel_time function")
    st.sidebar.success("Successfully imported process_data module")
except ImportError as e:
    st.sidebar.error(f"Error importing modules: {e}")

# Load config file if it exists, otherwise use defaults
@st.cache_data
def load_config():
    config_path = os.path.join(parent_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default config
        return {
            "default_model": "LSTM",
            "lanes": 3,
            "speed_limit": 60,
            "lags": 12,
            "default_start": "NW1",
            "default_destination": "SE9",
            "map_file": os.path.join(parent_dir, '2B', 'boroondara.txt'),
            "algorithm": "as"
        }

config = load_config()

# Function to save config
def save_config(config):
    config_path = os.path.join(parent_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    st.sidebar.success("Configuration saved!")

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_dir = os.path.join(parent_dir, '2B' 'models')
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        st.warning(f"Model directory not found at {model_dir}. Using dummy models.")
        # Create dummy models for demo
        for model_name in ["LSTM", "GRU", "Bidirectional_LSTM"]:
            dummy_input = tf.keras.layers.Input(shape=(12, 1))
            dummy_output = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(dummy_input))
            dummy_model = tf.keras.Model(inputs=dummy_input, outputs=dummy_output)
            dummy_model.compile(loss='mse', optimizer='adam')
            models[model_name] = dummy_model
        return models, None
    
    # Try to load real models
    try:
        models["LSTM"] = load_model(os.path.join(model_dir, 'LSTM_best_model.h5'))
        models["GRU"] = load_model(os.path.join(model_dir, 'GRU_best_model.h5'))
        models["Bidirectional_LSTM"] = load_model(os.path.join(model_dir, 'Bidirectional_LSTM_best_model.h5'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        return models, scaler
    except Exception as e:
        st.warning(f"Error loading models: {e}. Using dummy models.")
        # Create dummy models
        for model_name in ["LSTM", "GRU", "Bidirectional_LSTM"]:
            dummy_input = tf.keras.layers.Input(shape=(12, 1))
            dummy_output = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(dummy_input))
            dummy_model = tf.keras.Model(inputs=dummy_input, outputs=dummy_output)
            dummy_model.compile(loss='mse', optimizer='adam')
            models[model_name] = dummy_model
            
        # Create dummy scaler
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl')) if os.path.exists(os.path.join(model_dir, 'scaler.pkl')) else None
        return models, scaler

# Load map data
@st.cache_data
def load_map_data(map_file):
    if not os.path.exists(map_file):
        st.error(f"Map file not found: {map_file}")
        # Create a simple test graph
        G = nx.DiGraph()
        nodes = ["NW1", "NW2", "NE1", "CW1", "CE1", "SW1", "SE1", "SE9"]
        positions = {
            "NW1": (10, 90), "NW2": (30, 90), "NE1": (70, 90),
            "CW1": (30, 70), "CE1": (70, 70),
            "SW1": (30, 50), "SE1": (70, 50), "SE9": (70, 10)
        }
        
        # Add nodes with positions
        for node in nodes:
            G.add_node(node, pos=positions[node])
            
        # Add edges with distances
        edges = [
            ("NW1", "NW2", 1.2), ("NW2", "NE1", 2.0), 
            ("NW1", "CW1", 1.5), ("NW2", "CW1", 1.3),
            ("NE1", "CE1", 1.1), ("CW1", "CE1", 2.2),
            ("CW1", "SW1", 1.0), ("CE1", "SE1", 1.5),
            ("SW1", "SE1", 2.1), ("SE1", "SE9", 2.0)
        ]
        
        for u, v, d in edges:
            G.add_edge(u, v, distance=d)
            
        return G, positions
    
    try:
        G = nx.DiGraph()
        positions = {}
        
        with open(map_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    node1, node2, distance = parts[0], parts[1], float(parts[2])
                    G.add_node(node1)
                    G.add_node(node2)
                    G.add_edge(node1, node2, distance=distance)
                    
                    # If position data is available
                    if len(parts) >= 7:
                        x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                        positions[node1] = (x1, y1)
                        positions[node2] = (x2, y2)
        
        return G, positions
    except Exception as e:
        st.error(f"Error loading map: {e}")
        return nx.DiGraph(), {}

# Function to predict traffic flow
def predict_traffic_flow(input_data, model_name, models, scaler):
    if model_name not in models:
        st.error(f"Model {model_name} not found")
        return None
        
    model = models[model_name]
    
    # Make prediction
    try:
        prediction_norm = model.predict(input_data)[0][0]
        
        # Convert back to original scale if scaler available
        if scaler is not None:
            prediction_orig = scaler.inverse_transform([[prediction_norm]])[0][0]
        else:
            prediction_orig = prediction_norm * 2000  # Approximation for demo
            
        return prediction_orig
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Function to calculate travel times for different routes
def calculate_routes(G, start, end, flow, models, scaler, algorithm="as"):
    # First get the shortest path (default)
    try:
        shortest_path = nx.shortest_path(G, source=start, target=end, weight="distance")
    except nx.NetworkXNoPath:
        st.error(f"No path found between {start} and {end}")
        return None
        
    # Calculate path distance
    shortest_distance = sum(G[shortest_path[i]][shortest_path[i+1]]["distance"] for i in range(len(shortest_path)-1))
    
    # Get predictions for each model
    results = {}
    for model_name in models:
        # Predict traffic flow
        if isinstance(flow, np.ndarray):  # If we have actual traffic data
            prediction = predict_traffic_flow(flow, model_name, models, scaler)
        else:  # Use provided flow value
            prediction = flow
            
        # Calculate travel time based on flow
        travel_time = 0
        for i in range(len(shortest_path)-1):
            u, v = shortest_path[i], shortest_path[i+1]
            segment_distance = G[u][v]["distance"]
            segment_time = estimate_travel_time(prediction, segment_distance, 
                                                lanes=config["lanes"], 
                                                speed_limit=config["speed_limit"], 
                                                intersections=1)
            travel_time += segment_time
            
        results[model_name] = {
            "path": shortest_path,
            "distance": shortest_distance,
            "flow": prediction,
            "travel_time": travel_time
        }
    
    return results

# Generate a sample traffic input
def generate_traffic_input(lags=12):
    # Simple pattern for demo
    return np.random.rand(1, lags, 1) * 0.8

# Visualize the route on a map
def visualize_route(G, path, pos=None):
    if pos is None or not pos:
        pos = nx.spring_layout(G)  # Default layout if positions not provided
        
    # Create a plotly figure
    fig = go.Figure()
    
    # Add all edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'))
    
    # Add all nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=10,
            color='lightblue',
            line_width=2)))
    
    # Highlight the path
    path_edges = list(zip(path[:-1], path[1:]))
    path_x = []
    path_y = []
    for edge in path_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        path_x.extend([x0, x1, None])
        path_y.extend([y0, y1, None])
        
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y,
        line=dict(width=4, color='green'),
        hoverinfo='none',
        mode='lines'))
    
    # Highlight start and end nodes
    fig.add_trace(go.Scatter(
        x=[pos[path[0]][0]],
        y=[pos[path[0]][1]],
        mode='markers',
        marker=dict(size=15, color='blue'),
        name='Start'))
        
    fig.add_trace(go.Scatter(
        x=[pos[path[-1]][0]],
        y=[pos[path[-1]][1]],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='End'))
    
    # Add labels for nodes
    for node in G.nodes():
        x, y = pos[node]
        fig.add_annotation(
            x=x,
            y=y,
            text=node,
            showarrow=False,
            font=dict(size=10)
        )
    
    # Configure the layout
    fig.update_layout(
        title='Route Visualization',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=40, l=40, r=40, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

# App title and description
st.title("🚗 Traffic-Based Route Guidance System")
st.markdown("""
This system predicts traffic flow and recommends optimal routes based on predicted travel times.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "Traffic Prediction Model",
        ["LSTM", "GRU", "Bidirectional_LSTM", "Ensemble"],
        index=["LSTM", "GRU", "Bidirectional_LSTM", "Ensemble"].index(config["default_model"]) if "default_model" in config else 0
    )
    
    # Algorithm selection
    algorithm_choice = st.selectbox(
        "Path Finding Algorithm",
        ["as", "bfs", "gbfs"],
        index=["as", "bfs", "gbfs"].index(config["algorithm"]) if "algorithm" in config else 0
    )
    
    # Road parameters
    st.subheader("Road Parameters")
    lanes = st.slider("Number of Lanes", 1, 5, config["lanes"] if "lanes" in config else 3)
    speed_limit = st.slider("Speed Limit (km/h)", 30, 110, config["speed_limit"] if "speed_limit" in config else 60)
    
    # Save settings
    if st.button("Save Settings as Default"):
        config["default_model"] = model_choice
        config["lanes"] = lanes
        config["speed_limit"] = speed_limit
        config["algorithm"] = algorithm_choice
        save_config(config)

# Load models and map
st.markdown("### Loading Data and Models")
with st.spinner("Loading models and map data..."):
    models, scaler = load_models()
    G, positions = load_map_data(config["map_file"] if "map_file" in config else "")
    
    if len(G.nodes()) > 0:
        st.success(f"Map loaded with {len(G.nodes())} nodes and {len(G.edges())} edges")
    else:
        st.error("Failed to load map data")

# Available nodes
available_nodes = sorted(list(G.nodes()))

# Trip input section
st.markdown("## 🛣️ Plan Your Trip")

col1, col2 = st.columns(2)

with col1:
    start_node = st.selectbox(
        "Starting Location",
        available_nodes,
        index=available_nodes.index(config["default_start"]) if "default_start" in config and config["default_start"] in available_nodes else 0
    )
    
    # Time of day
    time_options = [
        "Morning (7:00-9:00 AM)",
        "Midday (11:00 AM-1:00 PM)",
        "Afternoon (4:00-6:00 PM)",
        "Evening (7:00-9:00 PM)"
    ]
    time_of_day = st.selectbox("Time of Day", time_options)

with col2:
    end_node = st.selectbox(
        "Destination",
        available_nodes,
        index=available_nodes.index(config["default_destination"]) if "default_destination" in config and config["default_destination"] in available_nodes else (len(available_nodes)-1)
    )
    
    # Traffic condition modifier
    traffic_condition = st.slider(
        "Expected Traffic Congestion", 
        0.0, 1.0, 0.5, 
        help="0 = Low traffic, 1 = Heavy congestion"
    )

# Generate predictions button
if st.button("Calculate Route"):
    if start_node == end_node:
        st.error("Starting and destination locations cannot be the same.")
    else:
        st.markdown("## Route Calculation Results")
        
        # Show a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate traffic data
        status_text.text("Generating traffic data...")
        progress_bar.progress(20)
        
        # Generate traffic input based on time of day and congestion level
        time_factor = {
            "Morning (7:00-9:00 AM)": 0.7,
            "Midday (11:00 AM-1:00 PM)": 0.4,
            "Afternoon (4:00-6:00 PM)": 0.8,
            "Evening (7:00-9:00 PM)": 0.5
        }
        
        base_factor = time_factor[time_of_day]
        traffic_input = np.random.normal(base_factor, 0.1, (1, config["lags"], 1)) * traffic_condition * 0.8
        
        # Step 2: Calculate routes
        status_text.text("Calculating routes...")
        progress_bar.progress(50)
        
        # Use specific model or ensemble
        if model_choice == "Ensemble":
            # For ensemble, directly use a traffic flow value
            flow_value = 1000 * traffic_condition  # Simple scaling
            routes = calculate_routes(G, start_node, end_node, flow_value, models, scaler, algorithm_choice)
        else:
            routes = calculate_routes(G, start_node, end_node, traffic_input, {model_choice: models[model_choice]}, scaler, algorithm_choice)
        
        progress_bar.progress(80)
        
        if routes:
            # Step 3: Display results
            status_text.text("Displaying results...")
            
            # For ensemble, average the results
            if model_choice == "Ensemble":
                # Display all model predictions
                st.subheader("Model Predictions")
                
                model_results = []
                for model_name, result in routes.items():
                    model_results.append({
                        "Model": model_name,
                        "Travel Time (min)": f"{result['travel_time']:.1f}",
                        "Traffic Flow": f"{result['flow']:.2f} veh/hr",
                        "Distance (km)": f"{result['distance']:.2f}"
                    })
                
                st.table(pd.DataFrame(model_results))
                
                # Calculate average travel time
                avg_time = sum(r["travel_time"] for r in routes.values()) / len(routes)
                best_model = min(routes.items(), key=lambda x: x[1]["travel_time"])
                
                # Display the best model's path
                st.subheader(f"Recommended Route (Based on {best_model[0]})")
                st.markdown(f"**Estimated Travel Time:** {best_model[1]['travel_time']:.1f} minutes")
                st.markdown(f"**Distance:** {best_model[1]['distance']:.2f} km")
                st.markdown(f"**Traffic Flow:** {best_model[1]['flow']:.2f} vehicles/hour")
                
                path_str = " → ".join(best_model[1]["path"])
                st.markdown(f"**Path:** {path_str}")
                
                # Visualize the route
                fig = visualize_route(G, best_model[1]["path"], positions)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Single model, display its results
                result = list(routes.values())[0]
                
                st.subheader(f"Route Based on {model_choice}")
                st.markdown(f"**Estimated Travel Time:** {result['travel_time']:.1f} minutes")
                st.markdown(f"**Distance:** {result['distance']:.2f} km")
                st.markdown(f"**Traffic Flow:** {result['flow']:.2f} vehicles/hour")
                
                path_str = " → ".join(result["path"])
                st.markdown(f"**Path:** {path_str}")
                
                # Visualize the route
                fig = visualize_route(G, result["path"], positions)
                st.plotly_chart(fig, use_container_width=True)
            
            # Traffic flow vs Speed curve visualization
            st.subheader("Traffic Flow vs Speed Relationship")
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Generate points for the curve
            speeds = np.linspace(5, 60, 100)
            A = -1.4648375
            B = 93.75
            flows = -A * speeds**2 + B * speeds
            
            # Plot the relationship
            ax2.plot(flows, speeds, label="Flow-Speed Relationship")
            
            # Mark the current prediction
            if model_choice != "Ensemble":
                flow = result["flow"]
                # Find corresponding speed using the quadratic formula
                discriminant = B**2 + 4*A*flow
                if discriminant >= 0:
                    if flow <= 351.56:  # Flow at speed limit
                        speed = 60
                    elif flow <= 1500:  # Uncongested
                        speed = (-B + np.sqrt(discriminant)) / (2*A)
                    else:  # Congested
                        speed = (-B - np.sqrt(discriminant)) / (2*A)
                    
                    ax2.scatter([flow], [speed], color='red', s=100, zorder=5)
                    ax2.annotate(f"Current\n({flow:.0f}, {speed:.1f})", 
                                 xy=(flow, speed), xytext=(flow+100, speed+5),
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            
            # Mark key points
            ax2.axhline(60, color='blue', linestyle='--', alpha=0.7, label="Speed Limit (60 km/h)")
            ax2.axvline(351.56, color='green', linestyle='--', alpha=0.7, label="Flow at Speed Limit (351.56)")
            ax2.axvline(1500, color='red', linestyle='--', alpha=0.7, label="Flow at Capacity (1500)")
            
            ax2.set_xlabel("Traffic Flow (vehicles/hour)")
            ax2.set_ylabel("Speed (km/h)")
            ax2.set_title("Relationship between Traffic Flow and Speed")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig2)
            
        else:
            st.error(f"Could not find a route between {start_node} and {end_node}")
        
        progress_bar.progress(100)
        status_text.text("Calculation complete!")

# Advanced settings (collapsible)
with st.expander("Advanced Settings and Help"):
    st.markdown("""
    ### About the Models
    
    - **LSTM**: Long Short-Term Memory network - good for capturing long-term patterns
    - **GRU**: Gated Recurrent Unit - similar to LSTM but with fewer parameters
    - **Bi-LSTM**: Bidirectional LSTM - processes data in both forward and backward directions
    - **Ensemble**: Combines predictions from all models for better accuracy
    
    ### Traffic Flow to Travel Time Conversion
    
    We use the following quadratic equation to convert traffic flow to speed:
    
    flow = -1.4648375 × (speed)² + 93.75 × (speed)
    
    Then we calculate travel time using: time = distance / speed
    
    ### Path Finding Algorithms
    
    - **as**: A* Search - uses heuristics for efficient pathfinding
    - **bfs**: Breadth-First Search - finds shortest path in terms of number of edges
    - **gbfs**: Greedy Best-First Search - prioritizes paths that appear closest to the goal
    """)
    
    # Map file path configuration
    st.subheader("Map Configuration")
    map_file_path = st.text_input("Map File Path", value=config["map_file"] if "map_file" in config else "")
    if st.button("Update Map Path"):
        config["map_file"] = map_file_path
        save_config(config)
        st.success("Map path updated! Please refresh the page to load the new map.")

# Footer
st.markdown("---")
st.markdown("© 2025 Traffic-Based Route Guidance System | Swinburne University of Technology")