import networkx as nx
from mercury.graph.core import Graph
from mercury.graph.ml import LeidenCommunities
import mercury.graph

# Create a simple test graph
G = nx.barbell_graph(10, 1)
g = Graph(G)

# Check if LeidenCommunities class is available
print(f"Leiden algorithm available: {hasattr(LeidenCommunities, '__init__')}")
print(f"Package version: {mercury.graph.__version__}")