"""
Compare Leiden and Louvain algorithms on various test networks.
This simplified version avoids GraphFrames dependencies.
"""

import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Simplified Leiden implementation (from previous file)
def leiden_community_detection(G, resolution=1.0, max_iterations=10):
    """
    Simple implementation of Leiden community detection using NetworkX for testing.
    """
    # Start with singleton partition (each node in its own community)
    partition = {node: i for i, node in enumerate(G.nodes())}
    
    # Track if improvement was made
    improvement = True
    iteration = 0
    
    # Track modularity
    modularity = calculate_modularity(G, partition, resolution)
    
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
        
        # If modularity decreased or no improvement, we're done
        if new_modularity <= modularity and not improvement:
            break
        
        modularity = new_modularity
    
    # Return final partition
    return partition, modularity

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

# Simplified Louvain implementation without the refinement phase
def louvain_community_detection(G, resolution=1.0, max_iterations=10):
    """
    Simple implementation of Louvain community detection using NetworkX for testing.
    """
    # Start with singleton partition (each node in its own community)
    partition = {node: i for i, node in enumerate(G.nodes())}
    
    # Track if improvement was made
    improvement = True
    iteration = 0
    
    # Track modularity
    modularity = calculate_modularity(G, partition, resolution)
    
    # Main loop
    while improvement and iteration < max_iterations:
        improvement = False
        iteration += 1
        
        # Create copy of partition
        new_partition = partition.copy()
        
        # Local moving phase (same as Leiden)
        for node in G.nodes():
            # Find best community for this node
            best_community = find_best_community(G, node, partition, resolution)
            
            # If improvement found, update partition
            if best_community != partition[node]:
                new_partition[node] = best_community
                improvement = True
        
        # Update partition 
        partition = new_partition
        
        # Calculate new modularity
        new_modularity = calculate_modularity(G, partition, resolution)
        
        # If modularity decreased or no improvement, we're done
        if new_modularity <= modularity and not improvement:
            break
        
        modularity = new_modularity
    
    # Return final partition
    return partition, modularity

def check_community_connectivity(G, communities):
    """Check if all communities are internally connected."""
    # Group nodes by community
    community_to_nodes = defaultdict(list)
    for node, comm in communities.items():
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

def run_comparison():
    """Run comparison tests between Leiden and Louvain."""
    # Test cases
    test_cases = {
        "Barbell Graph (2x10)": nx.barbell_graph(10, 1),
        "Karate Club": nx.karate_club_graph(),
        "Random Geometric Graph": nx.random_geometric_graph(100, 0.12, seed=42),
        "Watts-Strogatz Small World": nx.watts_strogatz_graph(100, 4, 0.1, seed=42)
    }
    
    # Results
    results = []
    
    # Run tests
    for name, G in test_cases.items():
        print(f"\n\nTesting on {name}")
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Run Leiden
        start = time.time()
        leiden_communities, leiden_modularity = leiden_community_detection(G, resolution=1.0)
        leiden_time = time.time() - start
        
        # Run Louvain
        start = time.time()
        louvain_communities, louvain_modularity = louvain_community_detection(G, resolution=1.0)
        louvain_time = time.time() - start
        
        # Check connectivity
        leiden_disconnected = check_community_connectivity(G, leiden_communities)
        louvain_disconnected = check_community_connectivity(G, louvain_communities)
        
        # Print results
        leiden_num_communities = len(set(leiden_communities.values()))
        louvain_num_communities = len(set(louvain_communities.values()))
        
        print(f"\nLeiden: {leiden_num_communities} communities, modularity={leiden_modularity:.4f}, time={leiden_time:.4f}s")
        print(f"Louvain: {louvain_num_communities} communities, modularity={louvain_modularity:.4f}, time={louvain_time:.4f}s")
        print(f"Leiden disconnected communities: {len(leiden_disconnected)}")
        print(f"Louvain disconnected communities: {len(louvain_disconnected)}")
        
        results.append({
            "name": name,
            "leiden_communities": leiden_num_communities,
            "leiden_modularity": leiden_modularity,
            "leiden_time": leiden_time,
            "leiden_disconnected": len(leiden_disconnected),
            "louvain_communities": louvain_num_communities,
            "louvain_modularity": louvain_modularity,
            "louvain_time": louvain_time,
            "louvain_disconnected": len(louvain_disconnected),
        })
    
    # Summary
    print("\n\n--- SUMMARY ---")
    print(f"{'Test Case':<25} {'Leiden Comm.':<12} {'Leiden Mod.':<12} {'Leiden Disc.':<12} {'Louvain Comm.':<13} {'Louvain Mod.':<12} {'Louvain Disc.':<12}")
    
    for result in results:
        print(f"{result['name']:<25} {result['leiden_communities']:<12} {result['leiden_modularity']:.4f}{' ':<6} {result['leiden_disconnected']:<12} {result['louvain_communities']:<13} {result['louvain_modularity']:.4f}{' ':<6} {result['louvain_disconnected']:<12}")

if __name__ == "__main__":
    run_comparison()