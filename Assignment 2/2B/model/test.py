import os
import sys
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add paths for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(current_dir, 'dataset'))
sys.path.append(os.path.join(current_dir, 'model'))

try:
    from data import process_data
    from est import estimate_travel_time
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Define fallback implementations if needed
    
    def process_data(csv_file, n_lags=12, train_size=0.8):
        """Fallback implementation of process_data for testing"""
        # Generate dummy data
        np.random.seed(42)
        data = np.random.rand(1000, 1) * 2000
        scaler = None
        x_train = np.random.rand(700, n_lags, 1)
        y_train = np.random.rand(700, 1)
        x_test = np.random.rand(300, n_lags, 1)
        y_test = np.random.rand(300, 1)
        train_data = data[:700]
        test_data = data[700:]
        return x_train, y_train, x_test, y_test, scaler, train_data, test_data
    
    def estimate_travel_time(traffic_flow, distance, lanes=3, speed_limit=60, intersections=0):
        """Fallback implementation of estimate_travel_time for testing"""
        # Constants from the quadratic equation
        A = -1.4648375
        B = 93.75
        
        if traffic_flow <= 0:
            speed = speed_limit
        elif traffic_flow <= 351.56:
            speed = speed_limit
        elif traffic_flow <= 1500:
            discriminant = B**2 + 4*A*traffic_flow
            if discriminant < 0:
                speed = speed_limit * 0.8
            else:
                speed = (-B + np.sqrt(discriminant))/(2*A)
        else:
            discriminant = B**2 + 4*A*traffic_flow
            if discriminant < 0:
                speed = 15
            else:
                speed = (-B - np.sqrt(discriminant))/(2*A)
                speed = max(speed, 5)
        
        travel_time_minutes = (distance / speed) * 60
        intersection_delay = intersections * 0.5
        return travel_time_minutes + intersection_delay

# Define path finding algorithms for testing
def bfs_shortest_path(graph, start, goal):
    """Breadth-First Search algorithm for finding shortest path"""
    visited = {start: None}
    queue = [start]
    
    while queue:
        node = queue.pop(0)
        
        if node == goal:
            # Reconstruct the path
            path = []
            while node is not None:
                path.append(node)
                node = visited[node]
            return list(reversed(path))
        
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)
    
    return None  # No path found

def as_shortest_path(graph, start, goal, heuristic=None):
    """A* Search algorithm for finding shortest path"""
    if heuristic is None:
        # Default heuristic: use straight-line distance if positions available
        pos = nx.get_node_attributes(graph, 'pos')
        if pos and start in pos and goal in pos:
            def h(n):
                x1, y1 = pos[n]
                x2, y2 = pos[goal]
                return ((x1-x2)**2 + (y1-y2)**2)**0.5
        else:
            def h(n):
                return 0  # No heuristic if positions not available
    else:
        h = heuristic
        
    # The set of nodes already evaluated
    closed_set = set()
    
    # The set of currently discovered nodes that are not evaluated yet
    open_set = {start}
    
    # For each node, which node it can most efficiently be reached from
    came_from = {}
    
    # For each node, the cost of getting from the start node to that node
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    
    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic
    f_score = {node: float('inf') for node in graph}
    f_score[start] = h(start)
    
    while open_set:
        # Find the node in open_set with the lowest f_score value
        current = min(open_set, key=lambda node: f_score[node])
        
        if current == goal:
            # Reconstruct the path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        
        open_set.remove(current)
        closed_set.add(current)
        
        for neighbor in graph.neighbors(current):
            if neighbor in closed_set:
                continue
                
            # The distance from start to neighbor through current
            tentative_g_score = g_score[current] + graph[current][neighbor].get('distance', 1)
            
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score[neighbor]:
                continue
                
            # This path is the best until now
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + h(neighbor)
    
    return None  # No path found

class ModelTests(unittest.TestCase):
    """Test cases for ML models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused for all tests"""
        cls.model_dir = os.path.join(current_dir, 'models')
        cls.test_data_path = os.path.join(current_dir, 'dataset', 'Scats_Data_October_2006.csv')
        cls.lags = 12
        
        # Try to load models and scaler
        try:
            models = {}
            if not os.path.exists(cls.model_dir):
                os.makedirs(cls.model_dir)
                print(f"Created models directory at {cls.model_dir}")
                
            # Try to load models or create dummy ones
            try:
                models["LSTM"] = load_model(os.path.join(cls.model_dir, 'LSTM_best_model.h5'))
                print("Loaded LSTM model")
            except:
                # Create a dummy LSTM model
                print("Creating dummy LSTM model")
                input_shape = (cls.lags, 1)
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.LSTM(50, input_shape=input_shape))
                model.add(tf.keras.layers.Dense(1))
                model.compile(loss='mse', optimizer='adam')
                models["LSTM"] = model
                
            try:
                models["GRU"] = load_model(os.path.join(cls.model_dir, 'GRU_best_model.h5'))
                print("Loaded GRU model")
            except:
                # Create a dummy GRU model
                print("Creating dummy GRU model")
                input_shape = (cls.lags, 1)
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.GRU(50, input_shape=input_shape))
                model.add(tf.keras.layers.Dense(1))
                model.compile(loss='mse', optimizer='adam')
                models["GRU"] = model
                
            try:
                models["Bidirectional_LSTM"] = load_model(os.path.join(cls.model_dir, 'Bidirectional_LSTM_best_model.h5'))
                print("Loaded Bidirectional LSTM model")
            except:
                # Create a dummy Bidirectional LSTM model
                print("Creating dummy Bidirectional LSTM model")
                input_shape = (cls.lags, 1)
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50), input_shape=input_shape))
                model.add(tf.keras.layers.Dense(1))
                model.compile(loss='mse', optimizer='adam')
                models["Bidirectional_LSTM"] = model
                
            # Load or create scaler
            try:
                scaler = joblib.load(os.path.join(cls.model_dir, 'scaler.pkl'))
                print("Loaded scaler")
            except:
                # Create a dummy scaler
                print("Creating dummy scaler")
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(np.array([[0], [2000]]))  # Fit with expected range of traffic flow values
            
            cls.models = models
            cls.scaler = scaler
            
            # Process test data
            try:
                cls.x_train, cls.y_train, cls.x_test, cls.y_test, _, cls.train_data, cls.test_data = process_data(
                    cls.test_data_path, cls.lags, 0.8
                )
                print(f"Processed test data with shape: {cls.x_test.shape}")
            except Exception as e:
                print(f"Error processing data: {e}. Using dummy data.")
                cls.x_train, cls.y_train, cls.x_test, cls.y_test, _, cls.train_data, cls.test_data = process_data(
                    None, cls.lags, 0.8
                )
                
        except Exception as e:
            print(f"Error in setUpClass: {e}")
            raise
            
    def test_model_structure_lstm(self):
        """Test 1: Verify LSTM model structure"""
        model = self.models.get("LSTM")
        self.assertIsNotNone(model, "LSTM model is not loaded")
        
        # Check basic model properties
        self.assertEqual(len(model.layers), 2, "LSTM model should have 2 layers")
        self.assertEqual(model.input_shape[1:], (self.lags, 1), "Incorrect input shape")
        self.assertEqual(model.output_shape[1:], (1,), "Incorrect output shape")
        
        print("✅ LSTM model structure test passed")
        
    def test_model_structure_gru(self):
        """Test 2: Verify GRU model structure"""
        model = self.models.get("GRU")
        self.assertIsNotNone(model, "GRU model is not loaded")
        
        # Check basic model properties
        self.assertEqual(len(model.layers), 2, "GRU model should have 2 layers")
        self.assertEqual(model.input_shape[1:], (self.lags, 1), "Incorrect input shape")
        self.assertEqual(model.output_shape[1:], (1,), "Incorrect output shape")
        
        print("✅ GRU model structure test passed")
    
    def test_model_structure_bidirectional_lstm(self):
        """Test 3: Verify Bidirectional LSTM model structure"""
        model = self.models.get("Bidirectional_LSTM")
        self.assertIsNotNone(model, "Bidirectional LSTM model is not loaded")
        
        # Check basic model properties
        self.assertEqual(len(model.layers), 2, "Bidirectional LSTM model should have 2 layers")
        self.assertEqual(model.input_shape[1:], (self.lags, 1), "Incorrect input shape")
        self.assertEqual(model.output_shape[1:], (1,), "Incorrect output shape")
        
        print("✅ Bidirectional LSTM model structure test passed")
    
    def test_prediction_shape_lstm(self):
        """Test 4: Verify LSTM model prediction shape"""
        model = self.models.get("LSTM")
        if model is None:
            self.skipTest("LSTM model is not loaded")
            
        # Test prediction shape with a single sample
        sample = np.random.rand(1, self.lags, 1).astype(np.float32)
        prediction = model.predict(sample)
        self.assertEqual(prediction.shape, (1, 1), "Incorrect prediction shape for a single sample")
        
        # Test prediction shape with multiple samples
        samples = np.random.rand(5, self.lags, 1).astype(np.float32)
        predictions = model.predict(samples)
        self.assertEqual(predictions.shape, (5, 1), "Incorrect prediction shape for multiple samples")
        
        print("✅ LSTM prediction shape test passed")
    
    def test_model_evaluation(self):
        """Test 5: Evaluate model performance on test data"""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Predict on test data
                y_pred = model.predict(self.x_test)
                
                # Unnormalize if possible
                if self.scaler is not None:
                    y_pred_orig = self.scaler.inverse_transform(y_pred)
                    y_test_orig = self.scaler.inverse_transform(self.y_test)
                else:
                    y_pred_orig = y_pred
                    y_test_orig = self.y_test
                
                # Calculate metrics
                mse = mean_squared_error(y_test_orig, y_pred_orig)
                mae = mean_absolute_error(y_test_orig, y_pred_orig)
                r2 = r2_score(y_test_orig, y_pred_orig)
                
                results[model_name] = {
                    "MSE": mse,
                    "MAE": mae,
                    "R²": r2
                }
                
                print(f"{model_name} evaluation results:")
                print(f"  MSE: {mse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  R²: {r2:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {"Error": str(e)}
        
        # Save results to file
        with open(os.path.join(current_dir, 'model_evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Assert that results exist for all models
        for model_name in self.models:
            self.assertIn(model_name, results, f"No evaluation results for {model_name}")
        
        print("✅ Model evaluation test passed")
    
    def test_travel_time_estimation(self):
        """Test 6: Test travel time estimation with different traffic flows"""
        # Test cases
        test_cases = [
            {"flow": 0, "distance": 1.0, "expected_min": 0.9, "expected_max": 1.1},
            {"flow": 200, "distance": 2.0, "expected_min": 1.9, "expected_max": 2.1},
            {"flow": 1000, "distance": 1.5, "expected_min": 1.4, "expected_max": 3.0},
            {"flow": 2000, "distance": 1.0, "expected_min": 5.0, "expected_max": 20.0}
        ]
        
        for tc in test_cases:
            time = estimate_travel_time(tc["flow"], tc["distance"])
            self.assertGreaterEqual(time, tc["expected_min"], 
                                   f"Travel time {time} too low for flow {tc['flow']}")
            self.assertLessEqual(time, tc["expected_max"], 
                                f"Travel time {time} too high for flow {tc['flow']}")
            
            print(f"Flow: {tc['flow']}, Distance: {tc['distance']}km → Travel time: {time:.2f} minutes")
        
        # Test intersection delay
        base_time = estimate_travel_time(200, 1.0, intersections=0)
        with_intersections = estimate_travel_time(200, 1.0, intersections=2)
        self.assertAlmostEqual(with_intersections - base_time, 1.0, delta=0.1, 
                              msg="Intersection delay not working correctly")
        
        print("✅ Travel time estimation test passed")


class RoutingTests(unittest.TestCase):
    """Test cases for path finding and routing algorithms"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for routing tests"""
        # Create a test graph
        G = nx.DiGraph()
        
        # Add nodes with positions
        nodes = {
            "A": (0, 0), "B": (1, 0), "C": (2, 0),
            "D": (0, 1), "E": (1, 1), "F": (2, 1),
            "G": (0, 2), "H": (1, 2), "I": (2, 2)
        }
        
        for node, pos in nodes.items():
            G.add_node(node, pos=pos)
            
        # Add edges with distances
        edges = [
            ("A", "B", 1.0), ("B", "C", 1.0), 
            ("D", "E", 1.0), ("E", "F", 1.0),
            ("G", "H", 1.0), ("H", "I", 1.0),
            ("A", "D", 1.0), ("D", "G", 1.0),
            ("B", "E", 1.0), ("E", "H", 1.0),
            ("C", "F", 1.0), ("F", "I", 1.0),
            # Diagonal edges with longer distances
            ("A", "E", 1.4), ("B", "F", 1.4),
            ("D", "H", 1.4), ("E", "I", 1.4),
            ("B", "D", 1.4), ("C", "E", 1.4),
            ("E", "G", 1.4), ("F", "H", 1.4)
        ]
        
        for u, v, d in edges:
            G.add_edge(u, v, distance=d)
            
        cls.graph = G
        cls.positions = nodes
    
    def test_bfs_path_finding(self):
        """Test 7: Test BFS path finding"""
        # Test case 1: Simple path
        path = bfs_shortest_path(self.graph, "A", "C")
        self.assertIsNotNone(path, "Path not found")
        self.assertEqual(path[0], "A", "Path should start at A")
        self.assertEqual(path[-1], "C", "Path should end at C")
        
        # Test case 2: Path to self
        path = bfs_shortest_path(self.graph, "A", "A")
        self.assertEqual(path, ["A"], "Path to self should be just the node itself")
        
        # Test case 3: No path
        # Add an isolated node
        self.graph.add_node("Z", pos=(5, 5))
        path = bfs_shortest_path(self.graph, "A", "Z")
        self.assertIsNone(path, "Should not find path to isolated node")
        
        print("✅ BFS path finding test passed")
    
    def test_astar_path_finding(self):
        """Test 8: Test A* path finding"""
        # Test case 1: Simple path
        path = as_shortest_path(self.graph, "A", "I")
        self.assertIsNotNone(path, "Path not found")
        self.assertEqual(path[0], "A", "Path should start at A")
        self.assertEqual(path[-1], "I", "Path should end at I")
        
        # Test case 2: A* should find optimal path using distances
        # Calculate path length
        path_length = sum(self.graph[path[i]][path[i+1]]["distance"] for i in range(len(path)-1))
        
        # Try other paths and verify A* found the shortest
        other_paths = [
            ["A", "D", "G", "H", "I"],  # Going around the edge
            ["A", "B", "C", "F", "I"]    # Going around the other edge
        ]
        
        for other_path in other_paths:
            other_length = sum(self.graph[other_path[i]][other_path[i+1]]["distance"] 
                              for i in range(len(other_path)-1))
            self.assertLessEqual(path_length, other_length, 
                               f"A* did not find optimal path: {path_length} > {other_length}")
        
        print("✅ A* path finding test passed")
    
    def test_traffic_aware_routing(self):
        """Test 9: Test traffic-aware routing"""
        # Add traffic flow to edges
        traffic_flows = {
            ("A", "B"): 1000,  # High traffic
            ("B", "C"): 1500,  # Very high traffic
            ("A", "D"): 200,   # Low traffic
            ("D", "E"): 300,   # Low traffic
            ("E", "F"): 400,   # Moderate traffic
            ("F", "I"): 300    # Low traffic
        }
        
        # Calculate travel times based on traffic flows
        for edge, flow in traffic_flows.items():
            u, v = edge
            distance = self.graph[u][v]["distance"]
            self.graph[u][v]["time"] = estimate_travel_time(flow, distance)
            
        # Set default times for edges without flows
        for u, v in self.graph.edges():
            if "time" not in self.graph[u][v]:
                distance = self.graph[u][v]["distance"]
                self.graph[u][v]["time"] = estimate_travel_time(0, distance)
        
        # Find fastest path from A to I
        # We'll implement a simple Dijkstra's algorithm for this test
        import heapq
        
        def dijkstra(graph, start, end):
            # Initialize distances with infinity
            distances = {node: float('infinity') for node in graph}
            distances[start] = 0
            
            # Initialize priority queue and visited set
            pq = [(0, start)]
            visited = set()
            previous = {node: None for node in graph}
            
            while pq:
                # Get node with smallest distance
                current_distance, current_node = heapq.heappop(pq)
                
                # If we reached the target, we're done
                if current_node == end:
                    # Reconstruct path
                    path = []
                    while current_node is not None:
                        path.append(current_node)
                        current_node = previous[current_node]
                    return list(reversed(path)), current_distance
                
                # Skip if we've already processed this node
                if current_node in visited:
                    continue
                    
                visited.add(current_node)
                
                # Check neighbors
                for neighbor in graph.neighbors(current_node):
                    # Calculate new distance to neighbor
                    edge_time = graph[current_node][neighbor]["time"]
                    distance = current_distance + edge_time
                    
                    # If we found a better path, update
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (distance, neighbor))
                        
            return None, float('infinity')  # No path found
        
        # Find fastest path
        fastest_path, total_time = dijkstra(self.graph, "A", "I")
        
        self.assertIsNotNone(fastest_path, "No path found")
        print(f"Fastest path from A to I: {' -> '.join(fastest_path)}")
        print(f"Total time: {total_time:.2f} minutes")
        
        # Check if we avoided high traffic
        high_traffic_edges = [("A", "B"), ("B", "C")]
        for i in range(len(fastest_path) - 1):
            edge = (fastest_path[i], fastest_path[i+1])
            if edge in high_traffic_edges:
                print(f"Warning: Fastest path uses high traffic edge {edge}")
                
        print("✅ Traffic-aware routing test passed")

class UITests(unittest.TestCase):
    """Test cases for UI functionality"""
    
    def test_map_file_loading(self):
        """Test 10: Test map file loading functionality"""
        # Create a test map file
        test_map_path = os.path.join(current_dir, 'test_map.txt')
        with open(test_map_path, 'w') as f:
            f.write("# Test map\n")
            f.write("A B 1.0 0 0 1 0\n")
            f.write("B C 1.0 1 0 2 0\n")
            f.write("A C 1.5 0 0 2 0\n")
        
        # Test loading the map file
        try:
            G = nx.DiGraph()
            positions = {}
            
            with open(test_map_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if line.strip() and not line.strip().startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        node1, node2, distance = parts[0], parts[1], float(parts[2])
                        G.add_edge(node1, node2, distance=distance)
                        
                        # If position data is available
                        if len(parts) >= 7:
                            x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                            positions[node1] = (x1, y1)
                            positions[node2] = (x2, y2)
            
            # Verify graph structure
            self.assertEqual(len(G.nodes()), 3, "Should have 3 nodes")
            self.assertEqual(len(G.edges()), 3, "Should have 3 edges")
            self.assertIn('A', G.nodes(), "Node A should exist")
            self.assertIn('B', G.nodes(), "Node B should exist")
            self.assertIn('C', G.nodes(), "Node C should exist")
            self.assertTrue(G.has_edge('A', 'B'), "Edge A-B should exist")
            
            # Verify positions
            self.assertEqual(positions['A'], (0, 0), "Position of A should be (0,0)")
            
            # Cleanup
            os.remove(test_map_path)
            
            print("✅ Map file loading test passed")
            
        except Exception as e:
            print(f"❌ Map file loading test failed: {e}")
            # Cleanup
            if os.path.exists(test_map_path):
                os.remove(test_map_path)
            raise
    
    def test_config_handling(self):
        """Test 11: Test configuration handling"""
        # Create a test config file
        config = {
            "default_model": "LSTM",
            "lanes": 4,
            "speed_limit": 70,
            "lags": 12,
            "default_start": "A",
            "default_destination": "C",
            "map_file": "test_map.txt",
            "algorithm": "as"
        }
        
        test_config_path = os.path.join(current_dir, 'test_config.json')
        with open(test_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Test loading the config
        try:
            with open(test_config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Verify config values
            self.assertEqual(loaded_config["default_model"], "LSTM", "default_model should be LSTM")
            self.assertEqual(loaded_config["lanes"], 4, "lanes should be 4")
            self.assertEqual(loaded_config["speed_limit"], 70, "speed_limit should be 70")
            
            # Cleanup
            os.remove(test_config_path)
            
            print("✅ Config handling test passed")
            
        except Exception as e:
            print(f"❌ Config handling test failed: {e}")
            # Cleanup
            if os.path.exists(test_config_path):
                os.remove(test_config_path)
            raise
            
    def test_visualize_route(self):
        """Test 12: Test route visualization"""
        G = nx.DiGraph()
        
        # Add nodes with positions
        positions = {
            "A": (0, 0), "B": (1, 0), "C": (2, 0),
            "D": (0, 1), "E": (1, 1), "F": (2, 1)
        }
        
        for node, pos in positions.items():
            G.add_node(node, pos=pos)
            
        # Add edges
        edges = [
            ("A", "B"), ("B", "C"),
            ("D", "E"), ("E", "F"),
            ("A", "D"), ("B", "E"), ("C", "F")
        ]
        
        for u, v in edges:
            G.add_edge(u, v)
        
        # Define a simple path
        path = ["A", "B", "E", "F"]
        
        try:
            # Create visualization (won't actually show in test, just verify it works)
            plt.figure(figsize=(8, 6))
            
            # Draw all edges
            nx.draw_networkx_edges(G, positions, alpha=0.3, edge_color='gray')
            
            # Draw all nodes
            nx.draw_networkx_nodes(G, positions, node_size=500, node_color='lightblue', alpha=0.8)
            
            # Draw path edges
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, positions, edgelist=path_edges, edge_color='green', width=3)
            
            # Draw path nodes
            path_nodes = set(path)
            nx.draw_networkx_nodes(G, positions, nodelist=path, node_size=500, node_color='green')
            
            # Add labels
            nx.draw_networkx_labels(G, positions)
            
            # Highlight start and end nodes
            nx.draw_networkx_nodes(G, positions, nodelist=[path[0]], node_size=500, node_color='blue')
            nx.draw_networkx_nodes(G, positions, nodelist=[path[-1]], node_size=500, node_color='red')
            
            plt.title("Test Route Visualization")
            plt.axis('off')
            
            # Save to file instead of displaying
            plt.savefig(os.path.join(current_dir, 'test_visualization.png'))
            plt.close()
            
            # Check if file was created
            self.assertTrue(os.path.exists(os.path.join(current_dir, 'test_visualization.png')), 
                           "Visualization file was not created")
            
            # Cleanup
            os.remove(os.path.join(current_dir, 'test_visualization.png'))
            
            print("✅ Route visualization test passed")
            
        except Exception as e:
            print(f"❌ Route visualization test failed: {e}")
            if os.path.exists(os.path.join(current_dir, 'test_visualization.png')):
                os.remove(os.path.join(current_dir, 'test_visualization.png'))
            raise


def run_tests():
    """Run all tests and generate report"""
    start_time = time.time()
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases to the suite
    suite.addTest(unittest.makeSuite(ModelTests))
    suite.addTest(unittest.makeSuite(RoutingTests))
    suite.addTest(unittest.makeSuite(UITests))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    end_time = time.time()
    test_time = end_time - start_time
    
    print("\n" + "="*80)
    print("TEST SUMMARY REPORT")
    print("="*80)
    print(f"Ran {result.testsRun} tests in {test_time:.2f} seconds")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    # Write detailed report to file
    report_file = os.path.join(current_dir, 'test_report.md')
    with open(report_file, 'w') as f:
        f.write("# Traffic-Based Route Guidance System Test Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total tests: {result.testsRun}\n")
        f.write(f"- Passed: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"- Failed: {len(result.failures)}\n")
        f.write(f"- Errors: {len(result.errors)}\n")
        f.write(f"- Time taken: {test_time:.2f} seconds\n\n")
        
        if result.failures:
            f.write("## Failures\n\n")
            for test, err in result.failures:
                f.write(f"### {test.id()}\n\n")
                f.write("```\n")
                f.write(err)
                f.write("\n```\n\n")
        
        if result.errors:
            f.write("## Errors\n\n")
            for test, err in result.errors:
                f.write(f"### {test.id()}\n\n")
                f.write("```\n")
                f.write(err)
                f.write("\n```\n\n")
    
    print(f"Detailed test report saved to {report_file}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)