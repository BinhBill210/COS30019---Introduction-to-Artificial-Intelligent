import sys
import math
from collections import deque
import heapq

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
    
    def __lt__(self, other):
        return self.state < other.state  # For tie-breaking in priority queues

class Graph:
    def __init__(self, filename):
        self.nodes = {}
        self.edges = {}
        self.origin = None
        self.destinations = []
        self.parse_file(filename)
    
    def parse_file(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.endswith(':'):
                    section = line[:-1]
                    continue
                
                if section == "Nodes":
                    # Parse node data: e.g., "1: (4,1)"
                    node_id, coords = line.split(':')
                    node_id = int(node_id.strip())
                    coords = coords.strip()[1:-1].split(',')
                    x, y = int(coords[0]), int(coords[1])
                    self.nodes[node_id] = (x, y)
                
                elif section == "Edges":
                    # Parse edge data: e.g., "(2,1): 4"
                    edge, cost = line.split(':')
                    edge = edge.strip()[1:-1].split(',')
                    from_node, to_node = int(edge[0]), int(edge[1])
                    cost = int(cost.strip())
                    
                    if from_node not in self.edges:
                        self.edges[from_node] = {}
                    self.edges[from_node][to_node] = cost
                
                elif section == "Origin":
                    self.origin = int(line.strip())
                
                elif section == "Destinations":
                    self.destinations = [int(dest.strip()) for dest in line.split(';')]
            
            print(f"Loaded graph with {len(self.nodes)} nodes, {sum(len(edges) for edges in self.edges.values())} edges")
            print(f"Origin: {self.origin}, Destinations: {self.destinations}")
        
        except Exception as e:
            print(f"Error parsing file: {e}")
            sys.exit(1)
    
    def get_neighbors(self, node_id):
        """Return list of neighboring nodes and costs in ascending order of node ID"""
        if node_id not in self.edges:
            return []
        return sorted([(to_node, cost) for to_node, cost in self.edges[node_id].items()])
    
    def is_goal(self, node_id):
        """Check if node is a destination"""
        return node_id in self.destinations
    
    def heuristic(self, node_id, dest_id=None):
        """Calculate Euclidean distance heuristic"""
        if dest_id is None:
            # If no specific destination provided, use minimum distance to any destination
            return min(self.heuristic(node_id, dest) for dest in self.destinations)
        
        x1, y1 = self.nodes[node_id]
        x2, y2 = self.nodes[dest_id]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class SearchAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.nodes_created = 0
    
    def solution_path(self, node):
        """Return the sequence of actions and states from the root to this node"""
        actions = []
        states = []
        current = node
        
        while current.parent:
            actions.append(current.action)
            states.append(current.state)
            current = current.parent
        
        # Add the initial state
        states.append(current.state)
        
        # Reverse to get path from root to goal
        actions.reverse()
        states.reverse()
        
        return actions, states, node.path_cost

    def output_solution(self, filename, method, goal_node):
        """Output solution in required format"""
        _, states, _ = self.solution_path(goal_node)
        
        print(f"{filename} {method}")
        print(f"{goal_node.state} {self.nodes_created}")
        print(" ".join(map(str, states)))


class DFS(SearchAlgorithm):
    """Depth-First Search"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a stack for DFS
        stack = [start_node]
        explored = set()
        
        while stack:
            # Pop from the end of the list (stack behavior)
            node = stack.pop()
            
            if node.state in explored:
                continue
                
            explored.add(node.state)
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in ascending order, but add them in reverse
            # so that smallest will be popped first
            for neighbor_state, step_cost in reversed(neighbors):
                if neighbor_state not in explored:
                    child = Node(
                        state=neighbor_state,
                        parent=node,
                        action=f"{node.state}->{neighbor_state}",
                        path_cost=node.path_cost + step_cost
                    )
                    self.nodes_created += 1
                    
                    if self.graph.is_goal(child.state):
                        return child
                    
                    stack.append(child)
        
        return None  # No solution found


class BFS(SearchAlgorithm):
    """Breadth-First Search"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a queue for BFS
        queue = deque([start_node])
        explored = set([start_node.state])  # Track explored states to avoid duplicates
        
        while queue:
            # Pop from the beginning of the queue (FIFO)
            node = queue.popleft()
            
            # No need to check if node.state is in explored since we check before adding to queue
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in ascending order
            for neighbor_state, step_cost in neighbors:
                if neighbor_state not in explored:
                    child = Node(
                        state=neighbor_state,
                        parent=node,
                        action=f"{node.state}->{neighbor_state}",
                        path_cost=node.path_cost + step_cost
                    )
                    self.nodes_created += 1
                    
                    if self.graph.is_goal(child.state):
                        return child
                    
                    explored.add(neighbor_state)  # Mark as explored when added to queue
                    queue.append(child)
        
        return None  # No solution found


class GBFS(SearchAlgorithm):
    """Greedy Best-First Search"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a priority queue for GBFS
        # Priority is the heuristic value (lower is better)
        frontier = [(self.graph.heuristic(start_node.state), id(start_node), start_node)]
        heapq.heapify(frontier)
        
        # Keep track of explored nodes
        explored = set()
        # Keep track of nodes in frontier for quick lookup
        frontier_states = {start_node.state}
        
        while frontier:
            # Get node with lowest heuristic value
            _, _, node = heapq.heappop(frontier)
            frontier_states.remove(node.state)
            
            if self.graph.is_goal(node.state):
                return node
                
            explored.add(node.state)
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in ascending order
            for neighbor_state, step_cost in neighbors:
                if neighbor_state not in explored and neighbor_state not in frontier_states:
                    child = Node(
                        state=neighbor_state,
                        parent=node,
                        action=f"{node.state}->{neighbor_state}",
                        path_cost=node.path_cost + step_cost
                    )
                    self.nodes_created += 1
                    
                    # Add child to frontier with heuristic as priority
                    h = self.graph.heuristic(child.state)
                    heapq.heappush(frontier, (h, id(child), child))
                    frontier_states.add(child.state)
        
        return None  # No solution found


class AS(SearchAlgorithm):
    """A* Search"""
    def search(self):
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a priority queue for A*
        # Priority is f(n) = g(n) + h(n) where g(n) is path cost and h(n) is heuristic
        frontier = [(self.graph.heuristic(start_node.state), id(start_node), start_node)]
        heapq.heapify(frontier)
        
        # Keep track of explored nodes and their costs
        explored = {}  # state -> cost
        # Keep track of nodes in frontier for quick lookup
        frontier_states = {start_node.state: start_node.path_cost}
        
        while frontier:
            # Get node with lowest f value
            _, _, node = heapq.heappop(frontier)
            frontier_states.pop(node.state, None)
            
            if self.graph.is_goal(node.state):
                return node
            
            # Only process if this is the cheapest path to this state
            if node.state in explored and explored[node.state] <= node.path_cost:
                continue
                
            explored[node.state] = node.path_cost
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in ascending order
            for neighbor_state, step_cost in neighbors:
                child_cost = node.path_cost + step_cost
                
                # Skip if we've found a better path to this neighbor
                if neighbor_state in explored and explored[neighbor_state] <= child_cost:
                    continue
                    
                # Skip if this node is in frontier with a better path
                if neighbor_state in frontier_states and frontier_states[neighbor_state] <= child_cost:
                    continue
                
                child = Node(
                    state=neighbor_state,
                    parent=node,
                    action=f"{node.state}->{neighbor_state}",
                    path_cost=child_cost
                )
                self.nodes_created += 1
                
                # Add child to frontier with f(n) = g(n) + h(n) as priority
                h = self.graph.heuristic(child.state)
                f = child_cost + h
                heapq.heappush(frontier, (f, id(child), child))
                frontier_states[child.state] = child_cost
        
        return None  # No solution found


class CUS1(SearchAlgorithm):
    """Custom Uninformed Search - Iterative Deepening Depth-First Search (IDDFS)"""
    def search(self, max_depth=30):
        """
        Iterative Deepening Depth-First Search
        Combines the benefits of DFS (space efficiency) and BFS (completeness)
        by doing a series of depth-limited searches with increasing depth limits.
        """
        for depth in range(1, max_depth + 1):
            # Initialize a new depth-limited search
            start_node = Node(self.graph.origin)
            self.nodes_created = 1 if depth == 1 else self.nodes_created  # Only count once
            
            # Use visited set to avoid cycles within the current depth-limited search
            visited = set()
            
            result = self._depth_limited_search(start_node, depth, visited)
            if result is not None:
                return result
                
        return None  # No solution found within max_depth
        
    def _depth_limited_search(self, node, limit, visited):
        """Recursive helper function for depth-limited search"""
        if self.graph.is_goal(node.state):
            return node
            
        if limit <= 0:
            return None
            
        visited.add(node.state)
        
        # Expand the node
        neighbors = self.graph.get_neighbors(node.state)
        
        # Process neighbors in ascending order
        for neighbor_state, step_cost in neighbors:
            if neighbor_state not in visited:
                child = Node(
                    state=neighbor_state,
                    parent=node,
                    action=f"{node.state}->{neighbor_state}",
                    path_cost=node.path_cost + step_cost
                )
                self.nodes_created += 1
                
                result = self._depth_limited_search(child, limit - 1, visited.copy())
                if result is not None:
                    return result
                    
        return None


class CUS2(SearchAlgorithm):
    """Custom Informed Search - Weighted A* Search"""
    def search(self, weight=2.0):
        """
        Weighted A* Search
        A variation of A* that puts more emphasis on the heuristic.
        f(n) = g(n) + weight * h(n)
        
        This can find solutions faster than A* but they might not be optimal.
        Higher weights lead to faster searches but potentially less optimal paths.
        """
        # Initialize
        start_node = Node(self.graph.origin)
        self.nodes_created = 1
        
        if self.graph.is_goal(start_node.state):
            return start_node
        
        # Use a priority queue with weighted heuristic
        h_start = self.graph.heuristic(start_node.state)
        f_start = start_node.path_cost + weight * h_start
        frontier = [(f_start, id(start_node), start_node)]
        heapq.heapify(frontier)
        
        # Keep track of explored nodes and their costs
        explored = {}  # state -> cost
        # Keep track of nodes in frontier for quick lookup
        frontier_states = {start_node.state: start_node.path_cost}
        
        while frontier:
            # Get node with lowest f value
            _, _, node = heapq.heappop(frontier)
            frontier_states.pop(node.state, None)
            
            if self.graph.is_goal(node.state):
                return node
            
            # Only process if this is the cheapest path to this state
            if node.state in explored and explored[node.state] <= node.path_cost:
                continue
                
            explored[node.state] = node.path_cost
            
            # Expand the node
            neighbors = self.graph.get_neighbors(node.state)
            
            # Process neighbors in ascending order
            for neighbor_state, step_cost in neighbors:
                child_cost = node.path_cost + step_cost
                
                # Skip if we've found a better path to this neighbor
                if neighbor_state in explored and explored[neighbor_state] <= child_cost:
                    continue
                    
                # Skip if this node is in frontier with a better path
                if neighbor_state in frontier_states and frontier_states[neighbor_state] <= child_cost:
                    continue
                
                child = Node(
                    state=neighbor_state,
                    parent=node,
                    action=f"{node.state}->{neighbor_state}",
                    path_cost=child_cost
                )
                self.nodes_created += 1
                
                # Add child to frontier with weighted f(n) = g(n) + weight*h(n)
                h = self.graph.heuristic(child.state)
                f = child_cost + weight * h
                heapq.heappush(frontier, (f, id(child), child))
                frontier_states[child.state] = child_cost
        
        return None  # No solution found


def run_search(filename, method):
    graph = Graph(filename)
    
    if method == "DFS":
        algorithm = DFS(graph)
    elif method == "BFS":
        algorithm = BFS(graph)
    elif method == "GBFS":
        algorithm = GBFS(graph)
    elif method == "AS":
        algorithm = AS(graph)
    elif method == "CUS1":
        algorithm = CUS1(graph)
    elif method == "CUS2":
        algorithm = CUS2(graph)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)
    
    print(f"Running {method} on {filename}...")
    result = algorithm.search()
    
    if result:
        algorithm.output_solution(filename, method, result)
    else:
        print(f"No solution found using {method} on {filename}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print("Methods: DFS, BFS, GBFS, AS, CUS1, CUS2")
        sys.exit(1)
    
    filename = sys.argv[1]
    method = sys.argv[2]
    
    run_search(filename, method)


if __name__ == "__main__":
    main()