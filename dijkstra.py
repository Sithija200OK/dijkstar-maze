# dijkstra.py
import heapq

def dijkstra(graph, start, goal):
    """
    Implements Dijkstra's algorithm to find the shortest path in a graph.
    
    Parameters:
    - graph: The graph where each node is a cell and the edges are the neighbors
    - start: The starting node (e.g., "(1, 1)")
    - goal: The goal node (e.g., "(6, 6)")
    
    Returns:
    - path: The shortest path from start to goal as a list of nodes
    - cost: The total cost to reach the goal from the start node
    """
    
    # Priority queue to store the nodes to explore, with their current distance
    queue = [(0, start)]  # (cost, node)
    # Dictionary to store the shortest distance to each node
    distances = {start: 0}
    # Dictionary to store the best path to each node
    previous_nodes = {start: None}
    
    while queue:
        # Get the node with the smallest distance (cost)
        current_cost, current_node = heapq.heappop(queue)
        
        # If we have reached the goal, stop searching
        if current_node == goal:
            break
        
        # Explore the neighbors of the current node
        for neighbor, move_cost in graph.get(current_node, {}).items():
            new_cost = current_cost + move_cost
            
            # If this path to the neighbor is better, update the information
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (new_cost, neighbor))
    
    # Reconstruct the path from start to goal
    path = []
    current_node = goal
    while current_node:
        path.insert(0, current_node)
        current_node = previous_nodes.get(current_node)
    
    # If there is no path found, return None
    if not path or path[0] != start:
        return None, float('inf')
    
    return path, distances.get(goal, float('inf'))