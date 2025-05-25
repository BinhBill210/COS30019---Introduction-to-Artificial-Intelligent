import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import argparse
import time
from tensorflow.keras.models import load_model
import argparse 
from search import Graph, BFS, GBFS, AS

# Add path to dataset modules
sys.path.append('dataset')
from data import process_data
sys.path.append('model')
from est import load_best_model, estimate_travel_time, predict_travel_times

class TrafficAwarePathFinder:
    """
    Class that integrates traffic prediction with path finding algorithms 
    from Assignment 2A to find optimal routes.
    """
    
    def __init__(self, map_file, model_name='LSTM', lanes=3, speed_limit=60):
        """
        Initialize the path finder with a map and traffic prediction model.
        
        Args:
            map_file: Path to the map file (in the format used in Assignment 2A)
            model_name: Name of the traffic prediction model to use
            lanes: Default number of lanes for roads
            speed_limit: Default speed limit in km/h
        """
        self.lanes = lanes
        self.speed_limit = speed_limit
        self.model_name = model_name
        self.map_file = map_file
        
        # Load the map
        self.graph, self.node_positions = self._load_map(map_file)
        
        # Load the traffic prediction model
        try:
            self.model = load_best_model(model_name)
            print(f"Loaded {model_name} model successfully")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
        # Process traffic data
        try:
            self.x_train, self.y_train, self.x_test, self.y_test, self.scaler, self.time_train, self.time_test = self._load_traffic_data()
            print("Loaded traffic data successfully")
        except Exception as e:
            print(f"Error loading traffic data: {e}")
            sys.exit(1)
    
    def _load_map(self, map_file):
        """
        Load map data from a file and create a graph.
        
        Args:
            map_file: Path to the map file
            
        Returns:
            graph: NetworkX graph representation of the map
            node_positions: Dictionary of node positions for visualization
        """
        graph = nx.DiGraph()
        node_positions = {}
        
        try:
            with open(map_file, 'r') as f:
                lines = f.readlines()
                
            # Parse map file (assuming format from Assignment 2A)
            # Format: node1 node2 distance
            for line in lines:
                if line.strip().startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 3:  # Node1 Node2 Distance [X1 Y1 X2 Y2]
                    node1, node2, distance = parts[0], parts[1], float(parts[2])
                    
                    # Add nodes and edge
                    graph.add_node(node1)
                    graph.add_node(node2)
                    graph.add_edge(node1, node2, distance=distance)
                    
                    # Store positions if available
                    if len(parts) >= 7:
                        x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                        node_positions[node1] = (x1, y1)
                        node_positions[node2] = (x2, y2)
            
            return graph, node_positions
        
        except Exception as e:
            print(f"Error loading map file: {e}")
            sys.exit(1)
    
    def _load_traffic_data(self, data_file='dataset/Scats_Data_October_2006.csv'):
        """
        Load and process traffic data.
        
        Args:
            data_file: Path to the traffic data file
            
        Returns:
            Processed traffic data
        """
        lags = 12  # Number of time lags to consider
        train_ratio = 0.75  # Ratio of training data
        
        return process_data(data_file, lags, train_ratio)
    
    def predict_traffic_flows(self, time_index=None):
        """
        Predict traffic flows using the loaded model.
        
        Args:
            time_index: Optional index into the test data
            
        Returns:
            Dictionary mapping road segments to predicted traffic flows
        """
        # If no specific time index is given, use the latest data point
        if time_index is None:
            time_index = len(self.x_test) - 1
        
        # Make sure time_index is within bounds
        time_index = max(0, min(time_index, len(self.x_test) - 1))
        
        # Get the data point for the specified time
        data_point = self.x_test[time_index:time_index+1]
        
        # Predict traffic flow
        flow_prediction = self.model.predict(data_point)
        
        # Convert to actual flow value
        flow_value = self.scaler.inverse_transform(flow_prediction)[0][0]
        
        # For simplicity, we'll assign the same flow to all edges for now
        # In a real system, you would have different predictions for different road segments
        flow_dict = {edge: flow_value for edge in self.graph.edges()}
        
        return flow_dict
    
    def update_edge_costs(self, traffic_flows):
        """
        Update edge costs based on predicted traffic flows.
        
        Args:
            traffic_flows: Dictionary mapping edges to traffic flows
        """
        for edge in self.graph.edges():
            node1, node2 = edge
            distance = self.graph[node1][node2]['distance']
            
            # Get traffic flow for this edge
            flow = traffic_flows.get(edge, 0)
            
            # Estimate travel time based on traffic flow
            travel_time = estimate_travel_time(flow, distance, self.lanes, self.speed_limit)
            
            # Update edge cost
            self.graph[node1][node2]['cost'] = travel_time
    
    def find_path(self, start, goal, algorithm='as'):
        """
        Find the optimal path from start to goal using the specified algorithm.
        
        Args:
            start: Starting node
            goal: Goal node
            algorithm: 'bfs', 'gbfs', or 'as' (A*)
            
        Returns:
            path: List of nodes in the optimal path
            cost: Total cost (travel time) of the path
            stats: Dictionary containing search statistics
        """
        # Convert the graph to the format expected by the search algorithm
        # Create a temporary graph file in the format expected by the Graph class
        temp_graph_filename = "temp_traffic_graph.txt"
        
        with open(temp_graph_filename, 'w') as f:
            f.write("Nodes:\n")
            node_id_map = {}  # Map node names to integers
            rev_node_id_map = {}  # Map integers back to node names
            
            # Assign integer IDs to nodes for the Graph class
            for i, node in enumerate(self.graph.nodes()):
                node_id = i + 1  # 1-based indexing
                node_id_map[node] = node_id
                rev_node_id_map[node_id] = node
                
                # Use node positions if available, otherwise use (0,0)
                pos = self.node_positions.get(node, (0, 0))
                f.write(f"{node_id}: ({pos[0]},{pos[1]})\n")
            
            f.write("\nEdges:\n")
            for u, v in self.graph.edges():
                # Use travel time as cost, rounded to nearest integer
                cost = round(self.graph[u][v]['cost'])
                f.write(f"({node_id_map[u]},{node_id_map[v]}): {cost}\n")
            
            f.write("\nOrigin:\n")
            f.write(f"{node_id_map.get(start, 1)}\n")  # Default to first node if not found
            
            f.write("\nDestinations:\n")
            f.write(f"{node_id_map.get(goal, 2)}\n")  # Default to second node if not found
        
        # Create the graph object
        graph = Graph(temp_graph_filename)
        
        # Create and run the appropriate search algorithm
        start_time = time.time()
        
        if algorithm.lower() == 'bfs':
            search_algo = BFS(graph)
        elif algorithm.lower() == 'gbfs':
            search_algo = GBFS(graph)
        elif algorithm.lower() == 'as':
            search_algo = AS(graph)
        else:
            os.remove(temp_graph_filename)
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'bfs', 'gbfs', or 'as'.")
        
        # Run the search
        result = search_algo.search()
        end_time = time.time()
        
        # Clean up temporary file
        os.remove(temp_graph_filename)
        
        if result:
            # Extract the path and cost
            actions, states, cost = search_algo.solution_path(result)
            
            # Convert state IDs back to original node names
            path = [rev_node_id_map.get(state, state) for state in states]
            
            # Create stats dictionary
            stats = {
                'algorithm': algorithm,
                'nodes_created': search_algo.nodes_created,
                'time_taken': end_time - start_time
            }
            
            return path, cost, stats
        else:
            return None, float('inf'), {'algorithm': algorithm, 'error': 'No path found'}
    
    def visualize_path(self, path, title="Traffic-Aware Optimal Path"):
        """
        Visualize the path on the map.
        
        Args:
            path: List of nodes in the path
            title: Title for the visualization
        """
        if not path or len(path) < 2:
            print("No valid path to visualize")
            return
        
        if not self.node_positions:
            print("No position data available for visualization")
            return
        
        # Create a new graph for visualization
        G = nx.DiGraph()
        
        # Add all nodes and edges
        for node, pos in self.node_positions.items():
            G.add_node(node, pos=pos)
        
        for u, v in self.graph.edges():
            if u in self.node_positions and v in self.node_positions:
                cost = self.graph[u][v]['cost']
                G.add_edge(u, v, weight=cost)
        
        # Prepare path edges
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Get positions for all nodes
        pos = nx.get_node_attributes(G, 'pos')
        
        plt.figure(figsize=(12, 10))
        
        # Draw all nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightgray')
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
        
        # Draw path
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=100, node_color='green')
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2.0, edge_color='green')
        
        # Draw start and goal nodes
        start_node = path[0]
        end_node = path[-1]
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_size=150, node_color='blue')
        nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_size=150, node_color='red')
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        # Add edge labels (travel time in minutes)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.1f} min" for u, v in path_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        
        plt.title(title)
        plt.axis('off')
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"path_visualization_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved as {filename}")

def main():
    parser = argparse.ArgumentParser(description='Traffic-Aware Path Finding')
    parser.add_argument('--map', type=str, required=True, help='Path to map file')
    parser.add_argument('--start', type=str, required=True, help='Starting node')
    parser.add_argument('--goal', type=str, required=True, help='Goal node')
    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'GRU', 'Bidirectional_LSTM'],
                        help='Model to use for traffic prediction')
    parser.add_argument('--algorithm', type=str, default='as', choices=['bfs', 'gbfs', 'as'],
                        help='Path finding algorithm to use (bfs, gbfs, as)')
    parser.add_argument('--time_index', type=int, default=None, 
                        help='Time index for traffic prediction (default: latest)')
    parser.add_argument('--visualize', action='store_true', help='Visualize the path')
    
    args = parser.parse_args()
    
    # Initialize the traffic-aware path finder
    path_finder = TrafficAwarePathFinder(args.map, args.model)
    
    # Predict traffic flows
    traffic_flows = path_finder.predict_traffic_flows(args.time_index)
    
    # Update edge costs based on predicted traffic flows
    path_finder.update_edge_costs(traffic_flows)
    
    # Find optimal path
    path, cost, stats = path_finder.find_path(args.start, args.goal, args.algorithm)
    
    # Print results
    if path:
        print(f"\n{'-'*50}")
        print(f"Optimal path from {args.start} to {args.goal} using {args.algorithm.upper()}:")
        print(f"{' -> '.join(map(str, path))}")
        print(f"Total travel time: {cost:.2f} minutes")
        print(f"\nSearch statistics:")
        print(f"Nodes created: {stats.get('nodes_created', 'N/A')}")
        print(f"Time taken: {stats.get('time_taken', 'N/A'):.4f} seconds")
        print(f"{'-'*50}")
        
        # Visualize the path if requested
        if args.visualize:
            path_finder.visualize_path(path, f"Traffic-Aware Path ({args.algorithm.upper()}) - {args.model} Model")
    else:
        print(f"No path found from {args.start} to {args.goal}")

if __name__ == "__main__":
    main()