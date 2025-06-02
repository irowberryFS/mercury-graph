"""
Simple demonstration of the Leiden community detection algorithm.
This version avoids GraphFrames dependencies to make testing easier.
"""

import networkx as nx
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# Define a simplified version of the Leiden algorithm using NetworkX
def leiden_community_detection(G, resolution=1.0, max_iterations=10):
    """
    Simple implementation of Leiden community detection using NetworkX for testing.
    
    Args:
        G (nx.Graph): NetworkX graph
        resolution (float): Resolution parameter
        max_iterations (int): Maximum number of iterations
        
    Returns:
        dict: Mapping of nodes to community IDs
    """
    # Start with singleton partition (each node in its own community)
    partition = {node: i for i, node in enumerate(G.nodes())}
    
    # Track if improvement was made
    improvement = True
    iteration = 0
    
    # Track modularity
    modularity = calculate_modularity(G, partition, resolution)
    
    print(f"Initial modularity: {modularity:.4f}")
    
    # Main loop
    while improvement and iteration < max_iterations:
        improvement = False
        iteration += 1
        
        # Create copy of partition
        new_partition = partition.copy()
        
        # Phase 1: Local moving
        for node in G.nodes():
            # Find best community for this node
            best_community = find_best_community(G, node, partition, resolution)
            
            # If improvement found, update partition
            if best_community != partition[node]:
                new_partition[node] = best_community
                improvement = True
        
        # Update partition
        partition = new_partition
        
        # Phase 2: Refinement (ensure connected communities)
        partition = refine_partition(G, partition)
        
        # Calculate new modularity
        new_modularity = calculate_modularity(G, partition, resolution)
        
        print(f"Iteration {iteration}: modularity = {new_modularity:.4f}")
        
        # If modularity decreased or no improvement, we're done
        if new_modularity <= modularity and not improvement:
            break
        
        modularity = new_modularity
    
    # Return final partition
    return partition

def find_best_community(G, node, partition, resolution):
    """Find the community that gives the highest modularity gain for a node."""
    current_community = partition[node]
    best_gain = 0.0
    best_community = current_community
    
    # Get all neighboring communities
    neighbor_communities = set()
    for neighbor in G.neighbors(node):
        neighbor_communities.add(partition[neighbor])
    
    # Also consider the current community
    neighbor_communities.add(current_community)
    
    # Calculate the gain for each community
    for community in neighbor_communities:
        gain = calculate_modularity_gain(G, node, current_community, community, partition, resolution)
        
        if gain > best_gain:
            best_gain = gain
            best_community = community
    
    return best_community

def calculate_modularity_gain(G, node, current_community, target_community, partition, resolution):
    """Calculate the modularity gain from moving a node to a new community."""
    if current_community == target_community:
        return 0.0
    
    # Calculate internal connections to current and target communities
    k_i = G.degree(node)
    
    # Sum of degrees in current community
    k_current = sum(G.degree(n) for n, comm in partition.items() if comm == current_community)
    
    # Sum of degrees in target community
    k_target = sum(G.degree(n) for n, comm in partition.items() if comm == target_community)
    
    # Internal connections to current community
    e_current = sum(1 for neighbor in G.neighbors(node) if partition[neighbor] == current_community)
    
    # Internal connections to target community
    e_target = sum(1 for neighbor in G.neighbors(node) if partition[neighbor] == target_community)
    
    # Total edges
    m = G.number_of_edges()
    
    # Calculate modularity gain
    gain = ((e_target - e_current) / (2 * m)) - (resolution * k_i * (k_target - k_current + k_i) / (2 * m * m))
    
    return gain

def calculate_modularity(G, partition, resolution):
    """Calculate the modularity of a partition."""
    # Group nodes by community
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    
    # Calculate modularity
    m = G.number_of_edges()
    q = 0.0
    
    for nodes in communities.values():
        # Skip empty communities
        if not nodes:
            continue
            
        # Create subgraph for this community
        subgraph = G.subgraph(nodes)
        
        # Calculate internal edges
        e_c = subgraph.number_of_edges()
        
        # Calculate sum of degrees
        degrees = sum(G.degree(node) for node in nodes)
        
        # Add contribution to modularity
        q += (e_c / m) - resolution * ((degrees / (2 * m)) ** 2)
    
    return q

def refine_partition(G, partition):
    """Ensure communities are connected by splitting disconnected communities."""
    # Group nodes by community
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    
    # Check for disconnected communities
    new_partition = partition.copy()
    next_community_id = max(partition.values()) + 1
    
    for comm, nodes in communities.items():
        # Skip singleton communities
        if len(nodes) <= 1:
            continue
            
        # Create subgraph for this community
        subgraph = G.subgraph(nodes)
        
        # Check if connected
        if not nx.is_connected(subgraph):
            # Split into connected components
            components = list(nx.connected_components(subgraph))
            
            # Keep the largest component with the original community ID
            largest_component = max(components, key=len)
            
            # Assign new community IDs to other components
            for component in components:
                if component != largest_component:
                    for node in component:
                        new_partition[node] = next_community_id
                    next_community_id += 1
    
    return new_partition

def test_barbell_graph():
    """
    Test Leiden algorithm on a barbell graph, which has obvious community structure
    (two densely connected communities with a single link between them).
    """
    print("Testing Leiden algorithm on barbell graph...")
    
    # Create a barbell graph (two complete graphs connected by a single edge)
    G = nx.barbell_graph(10, 1)
    
    # Run the algorithm
    start = time.time()
    communities = leiden_community_detection(G, resolution=1.0, max_iterations=10)
    leiden_time = time.time() - start
    
    # Print results
    num_communities = len(set(communities.values()))
    print(f"\nLeiden algorithm found {num_communities} communities in {leiden_time:.4f} seconds")
    
    # Check for disconnected communities
    print("\nChecking for disconnected communities...")
    disconnected = check_community_connectivity(G, communities)
    
    print(f"Disconnected communities: {len(disconnected)}")
    if disconnected:
        print("  Details:", disconnected)
    
    # Visualize the results (text-based)
    print("\nCommunities:")
    comm_counts = {}
    for comm in set(communities.values()):
        nodes = [node for node, c in communities.items() if c == comm]
        comm_counts[comm] = len(nodes)
        print(f"  Community {comm}: {len(nodes)} nodes - {nodes}")
    
    print(f"\nCommunity sizes: {comm_counts}")
    print("\nTest completed successfully.")

def check_community_connectivity(G, communities):
    """Check if all communities are internally connected."""
    # Group nodes by community
    community_to_nodes = {}
    for node, comm in communities.items():
        if comm not in community_to_nodes:
            community_to_nodes[comm] = []
        community_to_nodes[comm].append(node)
    
    disconnected = {}
    for comm, nodes in community_to_nodes.items():
        # Skip singleton communities
        if len(nodes) <= 1:
            continue
            
        # Create subgraph for this community
        subgraph = G.subgraph(nodes)
        
        # Check if connected
        if not nx.is_connected(subgraph):
            components = list(nx.connected_components(subgraph))
            disconnected[comm] = {
                'num_components': len(components),
                'components': [list(comp) for comp in components]
            }
    
    return disconnected

if __name__ == "__main__":
    test_barbell_graph()