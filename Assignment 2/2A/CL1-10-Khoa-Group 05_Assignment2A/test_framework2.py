import os
import subprocess # Keep for potential other uses, but not for running search.py
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import re
import sys # Needed for potential error exits

# --- IMPORTANT: Import classes from search.py ---
# Assuming search.py is in the same directory or accessible via PYTHONPATH
try:
    from search import Graph, Node, DFS, BFS, GBFS, AS, CUS1, CUS2, SearchAlgorithm # Import necessary components
except ImportError:
    print("Error: Could not import from search.py. Make sure it's in the same directory or accessible.")
    sys.exit(1)
# ---

# Test case generator (remains the same)
def create_test_case(filename, nodes, edges, origin, destinations):
    os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
    with open(filename, 'w') as f:
        # Write nodes
        f.write("Nodes:\n")
        for node_id, coords in nodes.items():
            f.write(f"{node_id}: ({coords[0]},{coords[1]})\n")
        f.write("\n")

        # Write edges
        f.write("Edges:\n")
        for (from_node, to_node), cost in edges.items():
            f.write(f"({from_node},{to_node}): {cost}\n")
        f.write("\n")

        # Write origin
        f.write("Origin:\n")
        f.write(f"{origin}\n")
        f.write("\n")

        # Write destinations
        f.write("Destinations:\n")
        f.write("; ".join(map(str, destinations)))

    print(f"Created test case: {filename}")

# Function to visualize a test case (remains largely the same, still parses file)
# Could be optimized to take graph data directly, but kept for simplicity here.
def visualize_graph(filename, output_file=None, show=False):
    """
    Creates a visualization of the graph defined in the input file.

    Args:
        filename: Path to the test case file
        output_file: Path to save the visualization image (optional)
        show: Whether to display the graph (default False)
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Visualize graph - file not found: {filename}")
        return None, None, None, None # Return None values on error

    # Parse nodes
    nodes = {}
    node_pattern = re.compile(r'(\d+): \((\d+),(\d+)\)')
    for match in node_pattern.finditer(content):
        node_id = int(match.group(1))
        x = int(match.group(2))
        y = int(match.group(3))
        nodes[node_id] = (x, y)

    # Parse edges
    edges = {}
    edge_pattern = re.compile(r'\((\d+),(\d+)\): (\d+)')
    for match in edge_pattern.finditer(content):
        from_node = int(match.group(1))
        to_node = int(match.group(2))
        cost = int(match.group(3))
        edges[(from_node, to_node)] = cost

    # Parse origin
    origin = None
    origin_pattern = re.compile(r'Origin:\s*(\d+)')
    origin_match = origin_pattern.search(content)
    if origin_match:
        origin = int(origin_match.group(1))

    # Parse destinations
    destinations = []
    dest_pattern = re.compile(r'Destinations:\s*([\d; ]+)')
    dest_match = dest_pattern.search(content)
    if dest_match:
        destinations = [int(d.strip()) for d in dest_match.group(1).split(';')]

    if not nodes:
        print(f"Warning: No nodes parsed from {filename}")
        return None, None, None, None

    # Create directed graph using NetworkX for visualization
    G = nx.DiGraph()

    # Add nodes with positions
    for node_id, pos in nodes.items():
        G.add_node(node_id, pos=pos)

    # Add edges with weights
    for (from_node, to_node), cost in edges.items():
        # Ensure nodes exist before adding edge (handles malformed files)
        if from_node in G and to_node in G:
            G.add_edge(from_node, to_node, weight=cost)
        else:
            print(f"Warning: Skipping edge ({from_node}, {to_node}) due to missing node(s) in file {filename}")


    if show or output_file:
        plt.figure(figsize=(12, 10))

        # Get positions for all nodes
        pos = nx.get_node_attributes(G, 'pos')

        # Check if pos is empty (can happen if nodes list was empty)
        if not pos:
            print(f"Warning: Cannot draw graph for {filename} - no node positions found.")
            plt.close()
            return nodes, edges, origin, destinations

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.8)

        # Highlight origin and destinations
        if origin in G:
            nx.draw_networkx_nodes(G, pos, nodelist=[origin], node_color='green',
                                  node_size=800, alpha=0.8, label='Origin')
        elif origin is not None:
             print(f"Warning: Origin node {origin} not found in graph {filename}")


        valid_destinations = [d for d in destinations if d in G]
        if valid_destinations:
            nx.draw_networkx_nodes(G, pos, nodelist=valid_destinations, node_color='red',
                                  node_size=800, alpha=0.8, label='Destination')
        invalid_destinations = [d for d in destinations if d not in G]
        if invalid_destinations:
             print(f"Warning: Destination node(s) {invalid_destinations} not found in graph {filename}")


        # Draw edges with weights
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='blue',
                              connectionstyle='arc3,rad=0.1', arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

        # Add title with file name
        plt.title(f"Graph: {os.path.basename(filename)}", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.legend()

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True) # Ensure dir exists
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved graph visualization to {output_file}")

        if show:
            plt.show()
        else:
            plt.close()

    return nodes, edges, origin, destinations


# Function to create several test cases (remains the same)
def create_test_cases():
    # Create test_cases directory if it doesn't exist
    os.makedirs("test_cases", exist_ok=True)
    os.makedirs("test_visualizations", exist_ok=True) # Ensure visualization dir exists

    # Test case 1: Basic path (same as example in the PDF)
    nodes = { 1: (4, 1), 2: (2, 2), 3: (4, 4), 4: (6, 3), 5: (5, 6), 6: (7, 5) }
    edges = { (2, 1): 4, (3, 1): 5, (1, 3): 5, (2, 3): 4, (3, 2): 5, (4, 1): 6, (1, 4): 6, (4, 3): 5, (3, 5): 6, (5, 3): 6, (4, 5): 7, (5, 4): 8, (6, 3): 7, (3, 6): 7 }
    filename = "test_cases/test1.txt"
    create_test_case(filename, nodes, edges, 2, [5, 4])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 2: Multiple destinations with different costs
    nodes = { 1: (1, 1), 2: (2, 3), 3: (4, 2), 4: (5, 5), 5: (3, 6), 6: (7, 4) }
    edges = { (1, 2): 3, (1, 3): 5, (2, 3): 2, (2, 4): 6, (2, 5): 4, (3, 4): 3, (3, 6): 4, (4, 5): 2, (4, 6): 3, (5, 4): 2, (6, 4): 3 }
    filename = "test_cases/test2.txt"
    create_test_case(filename, nodes, edges, 1, [4, 5, 6])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 3: One-way paths (directed edges)
    nodes = { 1: (1, 1), 2: (3, 2), 3: (5, 1), 4: (2, 4), 5: (4, 4), 6: (6, 3) }
    edges = { (1, 2): 3, (2, 3): 2, (3, 6): 4, (1, 4): 4, (4, 5): 3, (5, 6): 3, (5, 2): 2 }
    filename = "test_cases/test3.txt"
    create_test_case(filename, nodes, edges, 1, [6])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 4: Disconnected graph
    nodes = { 1: (1, 1), 2: (3, 2), 3: (5, 1), 4: (2, 4), 5: (4, 4), 6: (6, 3) }
    edges = { (1, 2): 3, (2, 3): 2, (4, 5): 3, (5, 6): 3 }
    filename = "test_cases/test4.txt"
    create_test_case(filename, nodes, edges, 1, [6])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 5: Large graph
    nodes = {}
    for i in range(1, 21): nodes[i] = (i % 5 * 2, i // 5 * 2)
    edges = {}
    for i in range(1, 21):
        if i % 5 != 0: edges[(i, i+1)] = 1
        if i <= 15: edges[(i, i+5)] = 1
    edges.update({(1, 7): 2, (2, 8): 2, (6, 12): 2, (11, 17): 2, (16, 20): 2, (1, 15): 8, (5, 16): 7, (10, 20): 6})
    filename = "test_cases/test5.txt"
    create_test_case(filename, nodes, edges, 1, [20])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 6: Equal cost paths
    nodes = { 1: (0, 0), 2: (1, 1), 3: (2, 0), 4: (1, -1), 5: (3, 1), 6: (3, -1) }
    edges = { (1, 2): 2, (1, 4): 2, (2, 3): 2, (4, 3): 2, (3, 5): 2, (3, 6): 2 }
    filename = "test_cases/test6.txt"
    create_test_case(filename, nodes, edges, 1, [5, 6])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 7: No solution
    nodes = { 1: (0, 0), 2: (1, 1), 3: (2, 0), 4: (3, 1) }
    edges = { (1, 2): 1, (2, 3): 1 }
    filename = "test_cases/test7.txt"
    create_test_case(filename, nodes, edges, 1, [4])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 8: Single node
    nodes = { 1: (0, 0) }
    edges = {}
    filename = "test_cases/test8.txt"
    create_test_case(filename, nodes, edges, 1, [1])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 9: Cyclic paths
    nodes = { 1: (0, 0), 2: (1, 1), 3: (2, 0), 4: (1, -1), 5: (3, 1) }
    edges = { (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 1): 1, (3, 5): 2 }
    filename = "test_cases/test9.txt"
    create_test_case(filename, nodes, edges, 1, [5])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 10: Complex topology
    nodes = { i: ((i-1)%4*2, (i-1)//4*2) for i in range(1, 17)} # Simpler grid layout
    edges = {}
    for i in range(1, 17):
        for j in range(i+1, 17):
            x1, y1 = nodes[i]; x2, y2 = nodes[j]
            distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
            if distance < 3:
                edges[(i, j)] = round(distance * 2)
                if (i + j) % 3 != 0: edges[(j, i)] = round(distance * 2)
    filename = "test_cases/test10.txt"
    create_test_case(filename, nodes, edges, 1, [16])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 11: Dense graph (Reduced complexity slightly for clarity)
    nodes = { i: ((i-1)%5*2, (i-1)//5*2) for i in range(1, 17) } # 16 nodes
    edges = {}
    for i in range(1, 17):
        for j in range(1, 17):
            if i != j:
                x1, y1 = nodes[i]; x2, y2 = nodes[j]
                distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                if distance <= 3.0: # Connect slightly further
                    edges[(i, j)] = round(distance * 1.5)
    filename = "test_cases/test11.txt"
    create_test_case(filename, nodes, edges, 1, [16])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

    # Test case 12: Simple graph with clear optimal path
    nodes = { 1: (0, 0), 2: (1, 1), 3: (2, 0), 4: (3, 1), 5: (4, 0) }
    edges = { (1, 2): 3, (2, 3): 2, (3, 4): 4, (4, 5): 1, (1, 3): 6, (2, 4): 5, (3, 5): 7 }
    filename = "test_cases/test12.txt"
    create_test_case(filename, nodes, edges, 1, [5])
    visualize_graph(filename, f"test_visualizations/{os.path.basename(filename).replace('.txt','.png')}")

# Function to visualize path (remains largely the same, still parses file)
# Could be optimized to take graph/node data directly.
def visualize_path(test_file, algorithm, path, output_file=None):
    """
    Visualizes a solution path on the graph.

    Args:
        test_file: Path to the test case file
        algorithm: Name of the algorithm
        path: List of node IDs (should be integers) in the path
        output_file: Path to save the visualization (optional)
    """
    # Parse the graph from file - needed for node positions and edges
    nodes, edges_dict, origin, destinations = visualize_graph(test_file) # Use the same parser

    if nodes is None: # Check if graph parsing failed
        print(f"Error: Cannot visualize path for {test_file}, graph data missing.")
        return

    # Ensure path nodes are integers
    try:
        path_int = [int(node) for node in path]
    except (ValueError, TypeError) as e:
        print(f"Error converting path nodes to integers for {test_file}, {algorithm}: {path}. Error: {e}")
        return

    # Create directed graph using NetworkX for visualization
    G = nx.DiGraph()
    for node_id, pos in nodes.items():
        G.add_node(node_id, pos=pos)
    for (from_node, to_node), cost in edges_dict.items():
        if from_node in G and to_node in G:
            G.add_edge(from_node, to_node, weight=cost)

    plt.figure(figsize=(12, 10))
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        print(f"Warning: Cannot draw path for {test_file} - no node positions.")
        plt.close()
        return

    # Draw all nodes and edges faintly
    nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.3, node_color='lightgray')
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.2, edge_color='gray',
                          connectionstyle='arc3,rad=0.1', arrowsize=15)

    # Highlight origin and destinations
    if origin in G:
        nx.draw_networkx_nodes(G, pos, nodelist=[origin], node_color='green', node_size=800, alpha=0.8, label='Origin')
    valid_destinations = [d for d in destinations if d in G]
    if valid_destinations:
        nx.draw_networkx_nodes(G, pos, nodelist=valid_destinations, node_color='red', node_size=800, alpha=0.8, label='Destination')

    # Extract path edges ensuring they exist in the graph data
    path_edges = []
    path_cost = 0
    valid_path = True
    for i in range(len(path_int)-1):
        u, v = path_int[i], path_int[i+1]
        if u in G and v in G: # Check if nodes exist in graph
             edge_tuple = (u, v)
             if edge_tuple in edges_dict: # Check if edge exists
                 path_edges.append(edge_tuple)
                 path_cost += edges_dict[edge_tuple]
             else:
                 print(f"Warning: Path edge ({u}, {v}) not found in edges for {test_file}, {algorithm}. Path visualization might be incomplete.")
                 valid_path = False
                 # Still add the edge visually, but maybe dashed? For now, just draw it.
                 # path_edges.append(edge_tuple) # Option to still draw it
        else:
            print(f"Warning: Node {u} or {v} not found in graph for {test_file}, {algorithm}. Path visualization incomplete.")
            valid_path = False


    # Highlight path nodes and edges
    valid_path_nodes = [p for p in path_int if p in G]
    nx.draw_networkx_nodes(G, pos, nodelist=valid_path_nodes, node_color='blue', node_size=800, alpha=0.6, label='Path Nodes')

    if path_edges: # Only draw if there are edges to draw
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, alpha=1.0,
                            edge_color='blue', connectionstyle='arc3,rad=0.1',
                            arrowsize=25, label='Path Edges')

        # Draw edge labels for path edges
        path_edge_labels = {edge: edges_dict.get(edge, '?') for edge in path_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=path_edge_labels, font_size=12, font_weight='bold')


    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Add title
    path_str = ' -> '.join(map(str, path_int))
    cost_str = f"{path_cost}" if valid_path else f"Incomplete ({path_cost})"
    plt.title(f"{os.path.basename(test_file)} - {algorithm}\nPath: {path_str}\nCost: {cost_str}",
              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.legend()

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True) # Ensure dir exists
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved path visualization to {output_file}")
    # else: # Avoid showing plots automatically unless specified
    #     plt.show()

    plt.close()


# --- Function to run tests (Rewritten to use imports) ---
def run_tests():
    # Ensure directories exist
    os.makedirs("test_cases", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("test_visualizations", exist_ok=True)
    os.makedirs("test_paths", exist_ok=True)
    os.makedirs("test_plots", exist_ok=True)

    # First create the test cases if they don't exist
    create_test_cases() # This also creates visualizations

    # Map algorithm names to their classes from search.py
    algorithm_classes = {
        "DFS": DFS,
        "BFS": BFS,
        "GBFS": GBFS,
        "AS": AS,
        "CUS1": CUS1,
        "CUS2": CUS2
    }
    algorithms = list(algorithm_classes.keys())
    test_files = sorted([os.path.join("test_cases", f) for f in os.listdir("test_cases") if f.endswith('.txt')])

    # Dictionary to store results
    results = defaultdict(dict)

    print("\n===== RUNNING TESTS (using imports) =====")
    for algo_name in algorithms:
        print(f"\nTesting {algo_name}...")
        AlgoClass = algorithm_classes[algo_name]

        for test_file in test_files:
            print(f"  Running on: {os.path.basename(test_file)}")
            start_time = time.time()
            error_msg = None
            goal_node_obj = None
            nodes_created = 0
            path = []
            path_cost = 0
            path_length = 0
            success = False

            try:
                # 1. Create Graph object from the test file
                graph = Graph(test_file)
                if not graph.origin or not graph.nodes:
                     raise ValueError(f"Graph could not be loaded correctly from {test_file}")


                # 2. Instantiate the algorithm
                algorithm_instance = AlgoClass(graph)

                # 3. Run the search
                goal_node_obj = algorithm_instance.search() # This returns the goal Node object or None

                # 4. Process results
                if goal_node_obj:
                    # Use the solution_path method from SearchAlgorithm base class
                    _, path, path_cost = algorithm_instance.solution_path(goal_node_obj)
                    nodes_created = algorithm_instance.nodes_created
                    path_length = len(path)
                    success = True
                    print(f"    Success! Goal: {goal_node_obj.state}, Nodes created: {nodes_created}, Path length: {path_length}, Cost: {path_cost:.2f}")

                    # Visualize the path
                    try:
                        vis_path_file = f"test_paths/{os.path.basename(test_file)}_{algo_name}_path.png"
                        visualize_path(test_file, algo_name, path, vis_path_file)
                    except Exception as viz_error:
                        print(f"    Path visualization error: {viz_error}")

                else:
                    # No solution found by the algorithm
                    print(f"    No solution found")
                    error_msg = 'No solution'
                    # Still record nodes created if the search ran
                    nodes_created = algorithm_instance.nodes_created if hasattr(algorithm_instance, 'nodes_created') else 0


            except FileNotFoundError:
                 error_msg = f"Test file not found: {test_file}"
                 print(f"    Error: {error_msg}")
            except Exception as e:
                error_msg = f"Exception during search: {e}"
                print(f"    Error: {error_msg}")
                # Optionally capture traceback:
                # import traceback
                # error_msg += "\n" + traceback.format_exc()

            end_time = time.time()
            execution_time = end_time - start_time

            # Store results
            results[test_file][algo_name] = {
                'goal': goal_node_obj.state if success else None,
                'nodes_created': nodes_created,
                'path_length': path_length if success else 0,
                'path_cost': path_cost if success else 0,
                'execution_time': execution_time,
                'path': path if success else [],
                'success': success,
                'error': error_msg # Store error message if any
            }
            if not success:
                 results[test_file][algo_name]['execution_time'] = execution_time # Ensure time is recorded even on failure


    # Generate comprehensive report (should work with the new results structure)
    generate_report(results, test_files, algorithms)

# --- Reporting and Plotting Functions (remain mostly the same) ---
# Minor adjustments might be needed if result keys changed significantly,
# but the provided structure ('nodes_created', 'path_length', 'execution_time', 'success')
# seems compatible with the existing report/plot functions.

def generate_report(results, test_files, algorithms):
    """
    Generates a comprehensive performance report for all algorithms.
    (Largely unchanged from original, assumes results dict structure is similar)
    """
    print("\n===== ALGORITHM PERFORMANCE REPORT =====")

    # Create metrics for each algorithm
    algo_metrics = {algo: {
        'success_count': 0,
        'total_nodes': 0,
        'total_path_length': 0,
        'total_path_cost': 0.0, # Added cost metric
        'total_time': 0,
        'success_tests': [],
        'min_nodes': float('inf'), 'max_nodes': 0,
        'min_path': float('inf'), 'max_path': 0,
        'min_cost': float('inf'), 'max_cost': 0.0, # Added cost metric
        'min_time': float('inf'), 'max_time': 0
    } for algo in algorithms}

    num_test_files = len(test_files)

    # Process results
    for test_file in test_files:
        test_basename = os.path.basename(test_file)
        for algo in algorithms:
            if test_file in results and algo in results[test_file]:
                result = results[test_file][algo]
                if result.get('success', False):
                    metrics = algo_metrics[algo]
                    metrics['success_count'] += 1
                    metrics['success_tests'].append(test_basename) # Use basename for cleaner list
                    metrics['total_nodes'] += result['nodes_created']
                    metrics['total_path_length'] += result['path_length']
                    metrics['total_path_cost'] += result['path_cost']
                    metrics['total_time'] += result['execution_time']

                    # Update min/max values
                    metrics['min_nodes'] = min(metrics['min_nodes'], result['nodes_created'])
                    metrics['max_nodes'] = max(metrics['max_nodes'], result['nodes_created'])
                    metrics['min_path'] = min(metrics['min_path'], result['path_length'])
                    metrics['max_path'] = max(metrics['max_path'], result['path_length'])
                    metrics['min_cost'] = min(metrics['min_cost'], result['path_cost'])
                    metrics['max_cost'] = max(metrics['max_cost'], result['path_cost'])
                    metrics['min_time'] = min(metrics['min_time'], result['execution_time'])
                    metrics['max_time'] = max(metrics['max_time'], result['execution_time'])

    # Calculate averages and handle zero success cases
    for algo in algorithms:
        metrics = algo_metrics[algo]
        count = metrics['success_count']
        if count > 0:
            metrics['avg_nodes'] = metrics['total_nodes'] / count
            metrics['avg_path'] = metrics['total_path_length'] / count
            metrics['avg_cost'] = metrics['total_path_cost'] / count
            metrics['avg_time'] = metrics['total_time'] / count
        else:
            # Set averages and mins to NaN if no successes
            metrics['avg_nodes'] = metrics['avg_path'] = metrics['avg_cost'] = metrics['avg_time'] = float('nan')
            metrics['min_nodes'] = metrics['min_path'] = metrics['min_cost'] = metrics['min_time'] = float('nan')
            # Max remains 0

    # Print summary
    print("\n===== SUMMARY STATISTICS =====")
    print(f"\nTested on {num_test_files} files.")
    print("\nSuccess Rate:")
    for algo in algorithms:
        rate = (algo_metrics[algo]['success_count'] / num_test_files * 100) if num_test_files > 0 else 0
        print(f"  {algo}: {algo_metrics[algo]['success_count']}/{num_test_files} ({rate:.1f}%)")

    print("\nAverage Nodes Created (on success):")
    for algo, metrics in algo_metrics.items():
        if not np.isnan(metrics['avg_nodes']):
            print(f"  {algo}: {metrics['avg_nodes']:.2f} (min: {metrics['min_nodes']}, max: {metrics['max_nodes']})")
        else: print(f"  {algo}: N/A (0 successes)")

    print("\nAverage Path Length (on success):")
    for algo, metrics in algo_metrics.items():
         if not np.isnan(metrics['avg_path']):
             print(f"  {algo}: {metrics['avg_path']:.2f} (min: {metrics['min_path']}, max: {metrics['max_path']})")
         else: print(f"  {algo}: N/A (0 successes)")

    print("\nAverage Path Cost (on success):") # Added Cost
    for algo, metrics in algo_metrics.items():
         if not np.isnan(metrics['avg_cost']):
             print(f"  {algo}: {metrics['avg_cost']:.2f} (min: {metrics['min_cost']:.2f}, max: {metrics['max_cost']:.2f})")
         else: print(f"  {algo}: N/A (0 successes)")

    print("\nAverage Execution Time (seconds, total):") # Clarified this is total time
    # Calculate average time over all tests, not just success
    total_times_all = {algo: sum(results[tf][algo].get('execution_time', 0) for tf in test_files if algo in results[tf]) for algo in algorithms}
    avg_times_all = {algo: total_times_all[algo] / num_test_files if num_test_files > 0 else 0 for algo in algorithms}
    for algo in algorithms:
        # Use min/max from successful runs for context, but average over all runs
        min_t = algo_metrics[algo]['min_time'] if not np.isnan(algo_metrics[algo]['min_time']) else 'N/A'
        max_t = algo_metrics[algo]['max_time'] if algo_metrics[algo]['max_time'] > 0 else 'N/A'
        min_t_str = f"{min_t:.6f}" if isinstance(min_t, float) else min_t
        max_t_str = f"{max_t:.6f}" if isinstance(max_t, float) else max_t
        print(f"  {algo}: {avg_times_all[algo]:.6f} (Successful min: {min_t_str}, max: {max_t_str})")


    # Generate individual test case reports
    print("\n===== INDIVIDUAL TEST CASE RESULTS =====")
    for test_file in test_files:
        print(f"\nResults for {os.path.basename(test_file)}:")
        for algo in algorithms:
            if test_file in results and algo in results[test_file]:
                result = results[test_file][algo]
                time_str = f"{result.get('execution_time', 0):.6f}s"
                if result.get('success', False):
                    print(f"  {algo}: Goal={result['goal']}, Nodes={result['nodes_created']}, Len={result['path_length']}, Cost={result['path_cost']:.2f}, Time={time_str}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    nodes_info = f"Nodes={result.get('nodes_created', 'N/A')}, " if result.get('nodes_created') is not None else ""
                    print(f"  {algo}: Failed - {error_msg}. {nodes_info}Time={time_str}")
            else:
                print(f"  {algo}: No results")

    # Create performance comparison plots
    create_performance_plots(algo_metrics, algorithms, results, test_files)


def create_performance_plots(algo_metrics, algorithms, results, test_files):
    """
    Creates comparison plots for algorithm performance metrics.
    (Largely unchanged, but added path cost plot and adjusted existing plots)
    """
    num_test_files = len(test_files)
    if num_test_files == 0:
        print("No test files found, skipping plot generation.")
        return

    # Set up plot style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        print("Seaborn style not found, using default.")
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    plot_dir = "test_plots"
    os.makedirs(plot_dir, exist_ok=True) # Ensure plot dir exists


    # Plotting function helper
    def plot_metric(metric_key, title, ylabel, filename, is_time=False, is_cost=False):
        plt.figure(figsize=(12, 8))
        values = []
        labels = []
        for algo in algorithms:
            metrics = algo_metrics[algo]
            if not np.isnan(metrics[metric_key]):
                 val = metrics[metric_key]
                 if is_time: val *= 1000 # Convert s to ms
                 values.append(val)
                 labels.append(f"{val:.1f}" if not is_time and not is_cost else f"{val:.2f}" + (" ms" if is_time else ""))
            else:
                 values.append(0) # Plot 0 for NaN averages
                 labels.append("N/A")


        bars = plt.bar(algorithms, values, color=colors)
        plt.title(title, fontsize=16)
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(rotation=15, ha='right') # Rotate labels slightly if long

        # Add values above bars
        for bar, label in zip(bars, labels):
            if label != "N/A":
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01, label,
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.close()

    # 1. Success Rate Plot (Adjusted)
    plt.figure(figsize=(12, 8))
    success_rates = [(algo_metrics[algo]['success_count'] / num_test_files * 100) if num_test_files > 0 else 0 for algo in algorithms]
    bars = plt.bar(algorithms, success_rates, color=colors)
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{rate:.1f}%",
                ha='center', va='bottom', fontweight='bold')
    plt.title('Algorithm Success Rate', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14); plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(0, 110); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "success_rate.png"), dpi=300); plt.close()


    # 2. Average Nodes Created Plot
    plot_metric('avg_nodes', 'Average Nodes Created (on success)', 'Average Nodes', 'avg_nodes.png')

    # 3. Average Path Length Plot
    plot_metric('avg_path', 'Average Path Length (on success)', 'Average Path Length', 'avg_path_length.png')

    # 4. Average Path Cost Plot (New)
    plot_metric('avg_cost', 'Average Path Cost (on success)', 'Average Cost', 'avg_path_cost.png', is_cost=True)

    # 5. Average Execution Time Plot (Adjusted to use overall average time)
    plt.figure(figsize=(12, 8))
    # Calculate average time over all tests, not just success
    total_times_all = {algo: sum(results[tf][algo].get('execution_time', 0) for tf in test_files if algo in results[tf]) for algo in algorithms}
    avg_times_all_ms = [(total_times_all[algo] / num_test_files * 1000) if num_test_files > 0 else 0 for algo in algorithms]
    bars = plt.bar(algorithms, avg_times_all_ms, color=colors)
    for bar, val in zip(bars, avg_times_all_ms):
         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f"{val:.2f} ms",
                  ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.title('Average Execution Time (All Runs)', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14); plt.ylabel('Time (milliseconds)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "avg_time_all.png"), dpi=300); plt.close()


    # 6. Normalized Comparison Plot (Adjusted to include Cost)
    plt.figure(figsize=(14, 10))
    # Find max values for normalization (handle potential NaNs/zeros)
    max_nodes = max(algo_metrics[algo]['avg_nodes'] for algo in algorithms if not np.isnan(algo_metrics[algo]['avg_nodes'])) if any(not np.isnan(m['avg_nodes']) for m in algo_metrics.values()) else 1
    max_path = max(algo_metrics[algo]['avg_path'] for algo in algorithms if not np.isnan(algo_metrics[algo]['avg_path'])) if any(not np.isnan(m['avg_path']) for m in algo_metrics.values()) else 1
    max_cost = max(algo_metrics[algo]['avg_cost'] for algo in algorithms if not np.isnan(algo_metrics[algo]['avg_cost'])) if any(not np.isnan(m['avg_cost']) for m in algo_metrics.values()) else 1
    # Use overall avg time for normalization
    max_time = max(avg_times_all_ms) / 1000 if any(t > 0 for t in avg_times_all_ms) else 1 # Use seconds for comparison base


    # Create normalized metrics (handle NaN) - Lower is better for all these
    norm_nodes = [(m['avg_nodes'] / max_nodes if not np.isnan(m['avg_nodes']) else 1) for m in algo_metrics.values()] # Assign 1 (worst) if NaN
    norm_path = [(m['avg_path'] / max_path if not np.isnan(m['avg_path']) else 1) for m in algo_metrics.values()]
    norm_cost = [(m['avg_cost'] / max_cost if not np.isnan(m['avg_cost']) else 1) for m in algo_metrics.values()]
    norm_time = [(avg_times_all_ms[i]/1000 / max_time if max_time > 0 else 0) for i, algo in enumerate(algorithms)] # Normalize overall time


    width = 0.20 # Adjusted width for 4 bars
    x = np.arange(len(algorithms))
    plt.bar(x - 1.5*width, norm_nodes, width, label='Nodes Created (Norm)', color='#3274A1')
    plt.bar(x - 0.5*width, norm_path, width, label='Path Length (Norm)', color='#E1812C')
    plt.bar(x + 0.5*width, norm_cost, width, label='Path Cost (Norm)', color='#3A923A') # Added Cost bar
    plt.bar(x + 1.5*width, norm_time, width, label='Exec Time (Norm)', color='#C0392B') # Added Time bar


    plt.xlabel('Algorithm', fontsize=14); plt.ylabel('Normalized Value (Lower is Better)', fontsize=14)
    plt.title('Normalized Algorithm Performance Comparison', fontsize=16)
    plt.xticks(x, algorithms, rotation=15, ha='right')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "normalized_comparison.png"), dpi=300); plt.close()


    # 7. Heatmap for Path Costs (More relevant than length for optimal search)
    test_basenames = [os.path.basename(tf).replace('.txt','') for tf in test_files]
    cost_data = np.full((len(algorithms), len(test_files)), np.nan) # Initialize with NaN
    for i, algo in enumerate(algorithms):
        for j, test_file in enumerate(test_files):
            if test_file in results and algo in results[test_file] and results[test_file][algo].get('success', False):
                cost_data[i, j] = results[test_file][algo]['path_cost']

    plt.figure(figsize=(16, 8))
    cmap = plt.cm.viridis_r # Reversed Viridis: lower cost = brighter color
    cmap.set_bad('lightgray') # Color for NaN (no solution/error)

    plt.imshow(cost_data, cmap=cmap, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Path Cost')
    plt.xlabel('Test Case', fontsize=14); plt.ylabel('Algorithm', fontsize=14)
    plt.title('Path Cost by Test Case and Algorithm', fontsize=16)
    plt.xticks(np.arange(len(test_basenames)), test_basenames, rotation=45, ha='right')
    plt.yticks(np.arange(len(algorithms)), algorithms)

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(test_files)):
            if not np.isnan(cost_data[i, j]):
                cost_val = cost_data[i, j]
                # Determine text color based on background brightness (simple threshold)
                # Normalize cost relative to the max cost found for text color decision
                max_cost_overall = np.nanmax(cost_data) if not np.all(np.isnan(cost_data)) else 1
                norm_cost_val = cost_val / max_cost_overall if max_cost_overall > 0 else 0
                text_color = "white" if norm_cost_val > 0.6 else "black"
                plt.text(j, i, f"{cost_val:.1f}", ha="center", va="center", color=text_color, fontsize=8)
            else:
                 plt.text(j, i, "Fail", ha="center", va="center", color="red", fontsize=8)


    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "path_cost_heatmap.png"), dpi=300); plt.close()

    # Create a text report (similar to original, added cost)
    report_file = os.path.join(plot_dir, "performance_report.txt")
    with open(report_file, "w") as f:
        f.write("=== ALGORITHM PERFORMANCE REPORT ===\n\n")
        f.write(f"Test Suite: {num_test_files} files ({', '.join(test_basenames)})\n\n")

        # Success rates
        f.write("Success Rates:\n")
        for algo in algorithms:
            rate = (algo_metrics[algo]['success_count'] / num_test_files * 100) if num_test_files > 0 else 0
            f.write(f"  {algo}: {algo_metrics[algo]['success_count']}/{num_test_files} ({rate:.1f}%)\n")
            # f.write(f"    Successful: {', '.join(algo_metrics[algo]['success_tests'])}\n\n") # Maybe too verbose

        # Metrics on Success
        f.write("\nPerformance Metrics (Average on Successful Runs):\n")
        header = f"{'Algorithm':<10} {'Avg Nodes':<12} {'Avg Length':<12} {'Avg Cost':<12} {'Avg Time (ms)':<15}"
        f.write(header + "\n" + "="*len(header) + "\n")
        for algo in algorithms:
            m = algo_metrics[algo]
            avg_time_ms_succ = (m['avg_time'] * 1000) if not np.isnan(m['avg_time']) else 'N/A'
            f.write(f"{algo:<10} {m['avg_nodes']:<12.2f} {m['avg_path']:<12.2f} {m['avg_cost']:<12.2f} {avg_time_ms_succ:<15.2f}\n".replace('nan', 'N/A'))


        # Overall Execution Time (Average over ALL runs)
        f.write("\nOverall Average Execution Time (All Runs):\n")
        header = f"{'Algorithm':<10} {'Avg Time (ms)':<15}"
        f.write(header + "\n" + "="*len(header) + "\n")
        for i, algo in enumerate(algorithms):
             f.write(f"{algo:<10} {avg_times_all_ms[i]:<15.3f}\n")


        # Best performers
        f.write("\n=== BEST PERFORMERS (AVERAGE ON SUCCESS) ===\n")
        def find_best(metric, lower_is_better=True):
             valid_metrics = {a: m[metric] for a, m in algo_metrics.items() if not np.isnan(m[metric])}
             if not valid_metrics: return "N/A", float('nan')
             best_val = min(valid_metrics.values()) if lower_is_better else max(valid_metrics.values())
             best_algos = [a for a, v in valid_metrics.items() if v == best_val]
             return ", ".join(best_algos), best_val

        best_nodes_algo, best_nodes_val = find_best('avg_nodes')
        best_len_algo, best_len_val = find_best('avg_path')
        best_cost_algo, best_cost_val = find_best('avg_cost')
        best_time_succ_algo, best_time_succ_val = find_best('avg_time') # Time on success

        f.write(f"Lowest Avg Nodes:    {best_nodes_algo} ({best_nodes_val:.2f})\n")
        f.write(f"Lowest Avg Length:   {best_len_algo} ({best_len_val:.2f})\n")
        f.write(f"Lowest Avg Cost:     {best_cost_algo} ({best_cost_val:.2f})\n")
        f.write(f"Lowest Avg Time (Success): {best_time_succ_algo} ({best_time_succ_val*1000:.2f} ms)\n")

        # Fastest overall (average on all runs)
        if any(t > 0 for t in avg_times_all_ms):
            min_overall_time = min(avg_times_all_ms)
            fastest_overall = [algorithms[i] for i, t in enumerate(avg_times_all_ms) if t == min_overall_time]
            f.write(f"Lowest Avg Time (All Runs):  {', '.join(fastest_overall)} ({min_overall_time:.2f} ms)\n")

        print(f"\nGenerated performance report: {report_file}")
        print(f"Generated plots in directory: {plot_dir}")


# Main function (remains the same)
if __name__ == "__main__":
    run_tests()