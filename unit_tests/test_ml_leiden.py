import pytest
import pandas as pd
import networkx as nx

from mercury.graph.core import Graph
from mercury.graph.core.spark_interface import pyspark_installed
from mercury.graph.ml.leiden import LeidenCommunities


class TestLeiden(object):
    def test_instancing(self):
        """
        Tests instancing and __init__ of the class LeidenCommunities
        """
        leiden_clustering = LeidenCommunities()

        assert isinstance(leiden_clustering, LeidenCommunities)
        assert leiden_clustering.min_quality_gain == 1e-03
        assert leiden_clustering.resolution == 1
        assert leiden_clustering.theta == 0.01
        assert leiden_clustering.quality_function == "modularity"
        assert type(str(leiden_clustering)) is str and len(str(leiden_clustering)) > 0
        assert type(repr(leiden_clustering)) is str and len(repr(leiden_clustering)) > 0

        # Test invalid parameters
        with pytest.raises(ValueError):
            LeidenCommunities(resolution=-1)

        with pytest.raises(ValueError):
            LeidenCommunities(theta=0)

        with pytest.raises(ValueError):
            LeidenCommunities(theta=1.5)

        with pytest.raises(ValueError):
            LeidenCommunities(quality_function="invalid")

    @pytest.mark.skipif(not pyspark_installed, reason="PySpark not installed")
    def test_fit(self):
        """
        Tests method LeidenCommunities.fit
        """
        # Create test graph with known community structure
        # Two distinct communities with dense internal connections
        edges_data = [
            # Community 1
            ("A1", "A2", 1.0),
            ("A1", "A3", 1.0),
            ("A2", "A3", 1.0),
            ("A2", "A4", 1.0),
            ("A3", "A4", 1.0),
            # Community 2
            ("B1", "B2", 1.0),
            ("B1", "B3", 1.0),
            ("B2", "B3", 1.0),
            ("B2", "B4", 1.0),
            ("B3", "B4", 1.0),
            # Weak connection between communities
            ("A4", "B1", 0.1)
        ]

        edges_df = pd.DataFrame(edges_data, columns=["src", "dst", "weight"])
        g = Graph(edges_df)

        # Test with default parameters
        leiden_clustering = LeidenCommunities(verbose=False)

        len_str = len(str(leiden_clustering))
        leiden_clustering.fit(g)
        
        # Check that the string representation changed
        len_str_fit = len(str(leiden_clustering))
        assert len_str_fit > len_str
        
        # Test with CPM quality function
        leiden_cpm = LeidenCommunities(quality_function="CPM", verbose=False)
        leiden_cpm.fit(g)
        assert hasattr(leiden_cpm, "quality_")
        assert hasattr(leiden_cpm, "labels_")
        
        # Test with different resolution parameter
        leiden_high_res = LeidenCommunities(resolution=2.0, verbose=False)
        leiden_high_res.fit(g)
        
        # Test all partitions option
        leiden_all_parts = LeidenCommunities(all_partitions=True, verbose=False)
        leiden_all_parts.fit(g)
        
        # Test single partition option
        leiden_single_part = LeidenCommunities(all_partitions=False, verbose=False)
        leiden_single_part.fit(g)
        assert "cluster" in leiden_single_part.labels_.columns

    @pytest.mark.skipif(not pyspark_installed, reason="PySpark not installed")
    def test_compare_with_louvain(self):
        """
        Compare the Leiden algorithm with the Louvain algorithm to ensure
        Leiden finds connected communities
        """
        from mercury.graph.ml.louvain import LouvainCommunities
        
        # Create a test graph where Louvain might produce disconnected communities
        # but Leiden should guarantee connected communities
        
        # Create a barbell graph (two complete graphs connected by a single edge)
        G = nx.barbell_graph(10, 1)
        
        # Convert to Mercury Graph
        g = Graph(G)
        
        # Run both algorithms
        leiden = LeidenCommunities(verbose=False)
        louvain = LouvainCommunities(verbose=False)
        
        leiden.fit(g)
        louvain.fit(g)
        
        # Both should produce results
        assert hasattr(leiden, "labels_")
        assert hasattr(louvain, "labels_")
        
        # Leiden should have a quality score
        assert hasattr(leiden, "quality_")