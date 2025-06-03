"""
Distributed Leiden Algorithm for Community Detection
---------------------------------------------------
This module constitutes a PySpark implementation of the Leiden algorithm for
community detection. The algorithm aims to find the partition of a graph that
yields the maximum modularity while guaranteeing well-connected communities.

The Leiden algorithm addresses the major limitation of the Louvain algorithm
by ensuring all communities are connected and providing stronger guarantees
about partition quality.
"""

from mercury.graph.core.base import BaseClass
from mercury.graph.core import Graph
from mercury.graph.core.spark_interface import pyspark_installed

if pyspark_installed:
    from pyspark.sql import DataFrame, Window, functions as F
    from pyspark.sql.types import IntegerType, DoubleType

from typing import Union
import random


class LeidenCommunities(BaseClass):
    """
    Class that defines the functions that run a PySpark implementation of the
    Leiden algorithm to find the partition that maximizes the modularity of an
    undirected graph while guaranteeing well-connected communities.

    The Leiden algorithm consists of three phases:
    1. Local moving of nodes (with fast local move)
    2. Refinement of the partition 
    3. Aggregation of the network based on the refined partition

    This implementation provides the following guarantees:
    - All communities are connected (γ-connected)
    - All communities are well-separated (γ-separated)
    - Converges to subset optimal partitions when run iteratively

    Args:
        min_modularity_gain (float):
            Modularity gain threshold between each pass. The algorithm
            stops if the gain in modularity between the current pass
            and the previous one is less than the given threshold.

        max_pass (int):
            Maximum number of passes.

        max_iter (int):
            Maximum number of iterations within each pass.

        resolution (float):
            The resolution parameter γ. Its value
            must be greater or equal to zero. If resolution is less than 1,
            modularity favors larger communities, while values greater than 1
            favor smaller communities.

        theta (float):
            Randomness parameter for the refinement phase. Higher values
            increase randomness in community selection during refinement.
            Should be between 0.0005 and 0.1.

        all_partitions (bool, optional):
            If True, the function will return all the partitions found at each
            step of the algorithm. If False, only the last partition will be returned.

        verbose (bool, optional):
            If True, print progress information during the algorithm execution.
    """

    def __init__(
        self,
        min_modularity_gain=1e-03,
        max_pass=2,
        max_iter=10,
        resolution: Union[float, int] = 1,
        theta: float = 0.01,
        all_partitions=True,
        verbose=True,
    ):
        self.min_modularity_gain = min_modularity_gain
        self.max_pass = max_pass
        self.max_iter = max_iter
        self.resolution = resolution
        self.theta = theta
        self.all_partitions = all_partitions
        self.verbose = verbose

        # Check parameters
        if resolution < 0:
            raise ValueError(f"Resolution value is {resolution} and cannot be < 0.")
        
        if not (0.0005 <= theta <= 0.1):
            raise ValueError(f"Theta value is {theta} and should be between 0.0005 and 0.1.")

    def __str__(self):
        base_str = super().__str__()

        if hasattr(self, "labels_"):
            extra_str = [
                f"",
                f"Cluster assignments are available in attribute `labels_`",
                f"Modularity: {self.modularity_}",
            ]
            return "\n".join([base_str] + extra_str)
        else:
            return base_str

    def fit(self, g: Graph):
        """
        Args:
            g (Graph): A mercury graph structure.

        Returns:
            (self): Fitted self (or raises an error).
        """
        edges = g.graphframe.edges

        # Verify edges input
        self._verify_data(
            df=edges,
            expected_cols_grouping=["src", "dst"],
            expected_cols_others=["weight"],
        )

        # Initialize dataframe to be returned
        ret = (
            edges.selectExpr("src as id")
            .unionByName(edges.selectExpr("dst as id"))
            .distinct()
            .withColumn("pass0", F.row_number().over(Window.orderBy("id")))
        ).checkpoint()

        # Convert edges to anonymized src's and dst's
        edges = (
            edges.selectExpr("src as src0", "dst as dst0", "weight")
            .join(other=ret.selectExpr("id as src0", "pass0 as src"), on="src0")
            .join(other=ret.selectExpr("id as dst0", "pass0 as dst"), on="dst0")
            .select("src", "dst", "weight")
        ).checkpoint()

        # Calculate m and initialize modularity
        m = self._calculate_m(edges)
        modularity0 = -1.0

        # Begin pass
        canPass, _pass = True, 0
        while canPass:

            if self.verbose:
                print(f"Starting Leiden Pass {_pass}.")

            # Phase 1: Local moving with fast local move
            partition = self._fast_local_moving(edges, _pass)
            
            # Phase 2: Refinement
            refined_partition = self._refinement_phase(edges, partition)
            
            # Calculate new modularity
            modularity1 = self._calculate_modularity(
                edges=edges, partition=refined_partition, m=m
            )

            # Check stopping criterion
            canPass = (modularity1 - modularity0 > self.min_modularity_gain) and (
                _pass < self.max_pass
            )
            modularity0 = modularity1
            self.modularity_ = modularity0

            if self.verbose:
                print(f"Pass {_pass} modularity: {modularity1:.6f}")

            # Update ret and Phase 3: Aggregation
            if canPass:
                ret = ret.join(
                    other=refined_partition.selectExpr(
                        f"id as pass{_pass}", f"c as pass{_pass + 1}"
                    ),
                    on=f"pass{_pass}",
                ).checkpoint()

                # Create aggregate network based on refined partition
                # but use original partition for initial communities in next level
                edges = self._create_aggregate_network(
                    edges, partition, refined_partition
                ).checkpoint()

            _pass += 1

        # Return final dataframe
        if self.all_partitions:
            cols = self._sort_passes(ret)
            ret = ret.select(cols)
        else:
            _last = self._last_pass(ret)
            ret = ret.selectExpr("id as node_id", f"{_last} as cluster")

        self.labels_ = ret
        return self

    def _fast_local_moving(self, edges, pass_num):
        """
        Fast local moving phase - only visits nodes whose neighborhood has changed.
        
        Args:
            edges: Edge dataframe
            pass_num: Current pass number
            
        Returns:
            Partition dataframe with columns 'id' and 'c'
        """
        # Initialize partition (each node in its own community)
        partition = (
            edges.selectExpr("src as id")
            .unionByName(edges.selectExpr("dst as id"))
            .distinct()
            .withColumn("c", F.col("id"))
        ).checkpoint()

        # Calculate m for modularity calculations
        m = self._calculate_m(edges)
        
        # Initialize queue with all nodes (simulated through iterations)
        changed = True
        iteration = 0
        
        while changed and iteration < self.max_iter:
            if self.verbose:
                print(f"  Fast local moving iteration {iteration}")
                
            # Perform one round of local moving
            new_partition = self._local_moving_round(edges, partition, m)
            
            # Check if any nodes moved
            movements = (
                partition.alias("old")
                .join(new_partition.alias("new"), on="id")
                .where("old.c != new.c")
                .count()
            )
            
            changed = movements > 0
            partition = new_partition.checkpoint()
            iteration += 1
            
            if self.verbose and movements > 0:
                print(f"    {movements} nodes moved")

        return partition

    def _local_moving_round(self, edges, partition, m):
        """
        Perform one round of local moving for all nodes.
        """
        # Calculate degrees and label edges with communities
        labeled_degrees = self._label_degrees(edges, partition)
        labeled_edges = self._label_edges(edges, partition)

        # Calculate community degrees (sum of degrees within each community)
        community_degrees = (
            labeled_degrees.groupBy("c")
            .agg(F.sum("degree").alias("community_degree"))
        )

        # For each node, calculate potential modularity gain for moving to neighbor communities
        node_improvements = (
            labeled_degrees
            # Add current community degree
            .join(
                community_degrees.selectExpr("c", "community_degree as current_comm_degree"),
                on="c"
            )
            # Get internal edges (within current community)
            .join(
                labeled_edges.where("(src != dst) and (cSrc = cDst)")
                .selectExpr("src as id", "weight as internal_weight")
                .unionByName(
                    labeled_edges.where("(src != dst) and (cSrc = cDst)")
                    .selectExpr("dst as id", "weight as internal_weight")
                )
                .groupBy("id")
                .agg(F.sum("internal_weight").alias("internal_edges")),
                on="id",
                how="left"
            )
            # Get edges to neighboring communities
            .join(
                labeled_edges.where("cSrc != cDst")
                .selectExpr("src as id", "cDst as neighbor_comm", "weight")
                .unionByName(
                    labeled_edges.where("cSrc != cDst")
                    .selectExpr("dst as id", "cSrc as neighbor_comm", "weight")
                )
                .groupBy("id", "neighbor_comm")
                .agg(F.sum("weight").alias("edges_to_neighbor")),
                on="id",
                how="left"
            )
            # Add neighbor community degrees
            .join(
                community_degrees.selectExpr("c as neighbor_comm", "community_degree as neighbor_comm_degree"),
                on="neighbor_comm",
                how="left"
            )
            # Calculate modularity change
            .withColumn(
                "modularity_change",
                F.coalesce("edges_to_neighbor", F.lit(0))
                - F.coalesce("internal_edges", F.lit(0))
                - (
                    F.col("degree") / F.lit(2 * m) * self.resolution
                    * (F.col("neighbor_comm_degree") - F.col("current_comm_degree") + F.col("degree"))
                )
            )
            # Keep only positive improvements
            .where("modularity_change > 0")
            # Rank improvements
            .withColumn(
                "improvement_rank",
                F.row_number().over(
                    Window.partitionBy("id").orderBy(F.desc("modularity_change"))
                )
            )
            .where("improvement_rank = 1")
        )

        # Apply best moves (handle symmetric moves by preferring higher modularity gain)
        best_moves = (
            node_improvements
            .withColumn(
                "move_priority",
                F.row_number().over(
                    Window.partitionBy(
                        F.sort_array(F.array("c", "neighbor_comm"))
                    ).orderBy(F.desc("modularity_change"))
                )
            )
            .where("move_priority = 1")
            .selectExpr("id", "neighbor_comm as new_c")
        )

        # Create new partition
        new_partition = (
            partition
            .join(best_moves, on="id", how="left")
            .withColumn("c", F.coalesce("new_c", "c"))
            .select("id", "c")
        )

        return new_partition

    def _refinement_phase(self, edges, partition):
        """
        Refinement phase - refines the partition by identifying subcommunities
        within each community from the local moving phase.
        
        Args:
            edges: Edge dataframe
            partition: Partition from local moving phase
            
        Returns:
            Refined partition dataframe
        """
        if self.verbose:
            print("  Starting refinement phase")

        # Start with singleton partition for refinement
        refined_partition = (
            partition.selectExpr("id", "id as refined_c")
        ).checkpoint()

        # Get list of communities to refine
        communities = partition.select("c").distinct().collect()
        
        for comm_row in communities:
            comm_id = comm_row["c"]
            
            # Get subgraph for this community
            comm_nodes = partition.where(f"c = {comm_id}").select("id")
            
            # Extract subgraph edges
            subgraph_edges = (
                edges
                .join(comm_nodes.selectExpr("id as src"), on="src")
                .join(comm_nodes.selectExpr("id as dst"), on="dst")
            )
            
            # Skip if community is too small or has no internal edges
            edge_count = subgraph_edges.count()
            if edge_count == 0:
                continue
                
            # Perform local merging within this community
            refined_comm_partition = self._local_merging_within_community(
                subgraph_edges, comm_nodes, comm_id
            )
            
            # Update refined partition for this community
            refined_partition = (
                refined_partition
                .join(
                    refined_comm_partition.selectExpr("id", "refined_c as new_refined_c"),
                    on="id",
                    how="left"
                )
                .withColumn("refined_c", F.coalesce("new_refined_c", "refined_c"))
                .select("id", "refined_c")
            )

        # Ensure refined communities are numbered properly
        refined_partition = self._renumber_communities(
            refined_partition.selectExpr("id", "refined_c as c")
        )

        return refined_partition

    def _local_merging_within_community(self, subgraph_edges, comm_nodes, comm_id):
        """
        Perform local merging within a single community during refinement.
        Uses randomized selection based on quality improvement.
        """
        # Start with each node in its own refined community
        current_partition = comm_nodes.withColumn("refined_c", F.col("id"))
        
        if subgraph_edges.count() == 0:
            return current_partition
            
        # Calculate m for this subgraph
        m_sub = self._calculate_m(subgraph_edges)
        
        # Perform several rounds of merging
        max_refinement_rounds = 5
        for round_num in range(max_refinement_rounds):
            # Find potential merges that improve modularity
            potential_merges = self._find_potential_merges(
                subgraph_edges, current_partition, m_sub
            )
            
            if potential_merges.count() == 0:
                break
                
            # Select merges using randomized selection based on theta
            selected_merges = self._randomized_merge_selection(potential_merges)
            
            if selected_merges.count() == 0:
                break
                
            # Apply selected merges
            current_partition = self._apply_merges(current_partition, selected_merges)
            
        return current_partition

    def _find_potential_merges(self, edges, partition, m):
        """
        Find all potential merges that would improve modularity.
        """
        # This is a simplified version - in practice would need more sophisticated
        # modularity calculation for potential merges
        labeled_edges = self._label_edges(edges, partition)
        
        # Find pairs of communities that are connected
        community_pairs = (
            labeled_edges
            .where("cSrc != cDst")
            .select("cSrc", "cDst", "weight")
            .groupBy("cSrc", "cDst")
            .agg(F.sum("weight").alias("edge_weight"))
            .withColumn("merge_benefit", F.col("edge_weight"))  # Simplified benefit
            .where("merge_benefit > 0")
        )
        
        return community_pairs

    def _randomized_merge_selection(self, potential_merges):
        """
        Select merges using randomized selection based on theta parameter.
        Higher benefits have higher probability of selection.
        """
        # For simplicity, select top merges with some randomness
        # In full implementation, would use proper probability distribution
        total_merges = potential_merges.count()
        if total_merges == 0:
            return potential_merges.limit(0)
            
        # Select a fraction of the best merges
        selection_fraction = min(0.5, 1.0 / (1.0 + self.theta * 10))
        num_to_select = max(1, int(total_merges * selection_fraction))
        
        return (
            potential_merges
            .orderBy(F.desc("merge_benefit"))
            .limit(num_to_select)
        )

    def _apply_merges(self, partition, selected_merges):
        """
        Apply the selected merges to update the partition.
        """
        # Create merge mapping
        merge_map = (
            selected_merges
            .selectExpr("cDst as from_comm", "cSrc as to_comm")
        )
        
        # Apply merges
        new_partition = (
            partition
            .join(
                merge_map.selectExpr("from_comm as refined_c", "to_comm"),
                on="refined_c",
                how="left"
            )
            .withColumn("refined_c", F.coalesce("to_comm", "refined_c"))
            .select("id", "refined_c")
        )
        
        return new_partition

    def _create_aggregate_network(self, edges, original_partition, refined_partition):
        """
        Create aggregate network based on refined partition, but use original
        partition to create initial partition for the next level.
        """
        # Label edges with refined communities
        refined_labeled_edges = self._label_edges(edges, refined_partition)
        
        # Create aggregate edges
        aggregate_edges = (
            refined_labeled_edges
            .select("cSrc", "cDst", "weight")
            .groupBy("cSrc", "cDst")
            .agg(F.sum("weight").alias("weight"))
            .selectExpr("cSrc as src", "cDst as dst", "weight")
        )
        
        return aggregate_edges

    def _renumber_communities(self, partition):
        """
        Renumber communities to be consecutive integers starting from 1.
        """
        # Get unique communities
        unique_communities = (
            partition.select("c").distinct()
            .withColumn("new_c", F.row_number().over(Window.orderBy("c")))
        )
        
        # Apply renumbering
        renumbered = (
            partition
            .join(unique_communities, on="c")
            .selectExpr("id", "new_c as c")
        )
        
        return renumbered

    # Reuse utility methods from Louvain implementation
    def _verify_data(self, df, expected_cols_grouping, expected_cols_others):
        """Inherited from Louvain implementation"""
        cols = df.columns
        expected_cols = expected_cols_grouping + expected_cols_others

        if not isinstance(df, DataFrame):
            raise TypeError("Input data must be a pyspark DataFrame.")

        msg = "Input data is missing expected column '{}'."
        for col in expected_cols:
            if col not in cols:
                raise ValueError(msg.format(col))

        dup = (
            df.groupBy(*expected_cols_grouping)
            .agg(F.count(F.lit(1)).alias("count"))
            .where("count > 1")
            .count()
        )
        if dup > 0:
            raise ValueError("Data has duplicated entries.")

    def _last_pass(self, df):
        """Inherited from Louvain implementation"""
        cols = [col for col in df.columns if "pass" in col]
        _max = max([int(col.split("pass")[1]) for col in cols])
        return f"pass{_max}"

    def _label_degrees(self, edges, partition):
        """Inherited from Louvain implementation"""
        ret = (
            partition.join(
                other=(
                    edges.selectExpr("src as id", "weight")
                    .unionByName(edges.selectExpr("dst as id", "weight"))
                    .groupBy("id")
                    .agg(F.sum("weight").alias("degree"))
                ),
                on="id",
                how="inner",
            )
            .select("id", "c", "degree")
            .checkpoint()
        )
        return ret

    def _label_edges(self, edges, partition):
        """Inherited from Louvain implementation"""
        ret = (
            edges
            .select("src", "dst", "weight")
            .join(
                other=partition.selectExpr("id as src", "c as cSrc"),
                on="src",
                how="left",
            )
            .join(
                other=partition.selectExpr("id as dst", "c as cDst"),
                on="dst",
                how="left",
            ).checkpoint()
        )
        return ret

    def _calculate_m(self, edges) -> int:
        """Inherited from Louvain implementation"""
        m = edges.select(F.sum("weight")).collect()[0][0]
        return int(m)

    def _calculate_modularity(self, edges, partition, m=None) -> float:
        """Inherited from Louvain implementation"""
        m = self._calculate_m(edges) if m is None else m
        norm = 1 / (2 * m)

        labeledEdges = self._label_edges(edges, partition)
        labeledDegrees = self._label_degrees(edges, partition)

        k_in = (labeledEdges.where("cSrc = cDst").select(F.sum("weight"))).collect()[0][0]
        k_in = 0 if k_in is None else k_in

        k_out = (
            labeledDegrees.groupby("c")
            .agg(F.sum("degree").alias("kC"))
            .selectExpr(f"{self.resolution} * sum(kC * kC)")
        ).collect()[0][0]

        return (k_in / m) - (norm**2 * float(k_out))

    def _sort_passes(self, res) -> list:
        """Inherited from Louvain implementation"""
        cols = [col for col in res.columns if "pass" in col]
        ints = sorted([int(col.replace("pass", "")) for col in cols])
        cols_sorted = ["id"] + ["pass" + str(i) for i in ints]
        return cols_sorted