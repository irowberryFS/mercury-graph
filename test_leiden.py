"""
Simple test script for the Leiden community detection algorithm.
"""

import networkx as nx
import pandas as pd
from mercury.graph.core import Graph  # Correct import path
from mercury.graph.ml import LeidenCommunities, LouvainCommunities
import time

def test_barbell_graph():
    """
    Test Leiden algorithm on a barbell graph, which has obvious community structure
    (two densely connected communities with a single link between them).
    """
    print("Testing Leiden vs Louvain on barbell graph...")
    
    # Create a barbell graph (two complete graphs connected by a single edge)
    G = nx.barbell_graph(10, 1)
    
    # Convert to Mercury Graph
    g = Graph(G)
    
    # Time both algorithms
    start = time.time()
    leiden = LeidenCommunities(max_pass=2, verbose=True)
    leiden.fit(g)
    leiden_time = time.time() - start
    
    start = time.time()
    louvain = LouvainCommunities(max_pass=2, verbose=True)
    louvain.fit(g)
    louvain_time = time.time() - start
    
    # Print results
    print(f"\nLeiden algorithm found communities with quality {leiden.quality_:.4f}")
    print(f"Leiden execution time: {leiden_time:.4f} seconds")
    
    print(f"\nLouvain algorithm found communities with modularity {louvain.modularity_:.4f}")
    print(f"Louvain execution time: {louvain_time:.4f} seconds")
    
    # Check for disconnected communities
    print("\nChecking for disconnected communities...")
    
    # Extract community assignments
    leiden_communities = leiden.labels_.toPandas().set_index('node_id')['cluster'].to_dict() if hasattr(leiden.labels_, 'toPandas') else leiden.labels_.set_index('node_id')['cluster'].to_dict()
    louvain_communities = louvain.labels_.toPandas().set_index('node_id')['cluster'].to_dict() if hasattr(louvain.labels_, 'toPandas') else louvain.labels_.set_index('node_id')['cluster'].to_dict()
    
    # Check connectivity
    leiden_disconnected = check_community_connectivity(G, leiden_communities)
    louvain_disconnected = check_community_connectivity(G, louvain_communities)
    
    print(f"Leiden disconnected communities: {len(leiden_disconnected)}")
    if leiden_disconnected:
        print("  Details:", leiden_disconnected)
    
    print(f"Louvain disconnected communities: {len(louvain_disconnected)}")
    if louvain_disconnected:
        print("  Details:", louvain_disconnected)
    
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