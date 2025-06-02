"""
Distributed Leiden Algorithm for Community Detection
---------------------------------------------------
This module constitutes a PySpark implementation of the Leiden algorithm for
community detection. The algorithm is an improvement over the Louvain algorithm
that ensures well-connected communities.
"""

from mercury.graph.core.base import BaseClass
from mercury.graph.core import Graph
from mercury.graph.core.spark_interface import pyspark_installed

if pyspark_installed:
    from pyspark.sql import DataFrame, Window, functions as F

from typing import Union


class LeidenCommunities(BaseClass):
    """
    Class that defines the functions that run a PySpark implementation of the
    Leiden algorithm to find the partition that maximizes the modularity of an
    undirected graph with guaranteed well-connected communities.

    The Leiden algorithm is an improvement over the Louvain algorithm that addresses
    a major flaw in Louvain: the creation of badly connected or disconnected communities.
    Leiden guarantees that all communities are well-connected, which is particularly
    important for large complex networks.

    The key innovations over Louvain are:
    1. A fast local move phase that optimizes node reassignments efficiently
    2. A refinement phase that ensures communities are well-connected
    3. An aggregation phase based on the refined partition

    Contributed based on the work of Traag, V.A., Waltman, L. & van Eck, N.J.

    References:
        Traag, V.A., Waltman, L. & van Eck, N.J. "From Louvain to Leiden: 
        guaranteeing well-connected communities". Sci Rep 9, 5233 (2019).
        https://doi.org/10.1038/s41598-019-41695-z

    Args:
        min_quality_gain (float):
            Quality gain threshold between each pass. The algorithm
            stops if the gain in quality between the current pass
            and the previous one is less than the given threshold.

        max_pass (int):
            Maximum number of passes.

        max_iter (int):
            Maximum number of iterations within each pass.

        resolution (float):
            The resolution parameter gamma. Its value
            must be greater or equal to zero. If resolution is less than 1,
            quality optimization favors larger communities, while values greater 
            than 1 favor smaller communities.

        theta (float):
            Randomness parameter for the refinement phase. Controls the
            randomization in community mergers during refinement. Values should
            be between 0 and 1, with 0.01 as a reasonable default.

        quality_function (str):
            Quality function to use for optimization. Either "modularity" 
            (default) or "CPM" (Constant Potts Model).

        all_partitions (bool, optional):
            If True, the function will return all the partitions found at each
            step of the algorithm (i.e., pass0, pass1, pass2, ...). If
            False, only the last (and best) partition will be returned.

        verbose (bool, optional):
            If True, print progress information during the algorithm
            execution. Defaults to True.
    """

    def __init__(
        self,
        min_quality_gain=1e-03,
        max_pass=2,
        max_iter=10,
        resolution: Union[float, int] = 1,
        theta: float = 0.01,
        quality_function: str = "modularity",
        all_partitions=True,
        verbose=True,
    ):
        self.min_quality_gain = min_quality_gain
        self.max_pass = max_pass
        self.max_iter = max_iter
        self.resolution = resolution
        self.theta = theta
        self.quality_function = quality_function
        self.all_partitions = all_partitions
        self.verbose = verbose

        # Check resolution
        if resolution < 0:
            exceptionMsg = f"Resolution value is {resolution} and cannot be < 0."
            raise ValueError(exceptionMsg)
            
        # Check theta
        if theta <= 0 or theta > 1:
            exceptionMsg = f"Theta value is {theta} and must be in range (0, 1]."
            raise ValueError(exceptionMsg)
            
        # Check quality function
        if quality_function not in ["modularity", "CPM"]:
            exceptionMsg = f"Quality function must be 'modularity' or 'CPM', not {quality_function}."
            raise ValueError(exceptionMsg)

    def __str__(self):
        base_str = super().__str__()

        # Check if the object has been fitted (fitting creates the `labels_` attribute)
        if hasattr(self, "labels_"):
            extra_str = [
                f"",
                f"Cluster assignments are available in attribute `labels_`",
                f"Quality: {self.quality_:.6f}",
            ]
            return "\n".join([base_str] + extra_str)
        else:
            return base_str

    def fit(self, g: Graph):
        """
        Applies the Leiden algorithm to find communities in the graph.

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

        # Init dataframe to be returned
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

        # Calculate m and initialize quality
        m = self._calculate_m(edges)
        quality0 = -1.0

        # Begin pass
        canPass, _pass = True, 0
        while canPass:

            # Declare naive partition
            p1 = (
                edges.selectExpr("src as id")
                .unionByName(edges.selectExpr("dst as id"))
                .distinct()
                .withColumn("c", F.col("id"))
            )

            # Begin iterations within pass
            canIter, _iter = True, 0
            # Carry reference to previously cached p2 to call unpersist()
            prev_p2 = None
            while canIter:

                if _iter >= self.max_iter:
                    break

                # Print progress
                if self.verbose:
                    print(f"Starting Pass {_pass} Iteration {_iter}.")

                # Phase 1: Fast local move 
                p2 = self._fast_local_move(edges, p1, m=m)
                # Break complex lineage caused by loops first
                p2 = p2.checkpoint()
                p2.cache()

                canIter = len(p2.where("cx != cj").take(1)) > 0
                if canIter:
                    p1 = p2.selectExpr("id", "cj as c")
                if prev_p2 is not None:
                    prev_p2.unpersist()
                prev_p2 = p2
                _iter += 1

            # Phase 2: Refinement phase
            refined_p1 = self._refine_partition(edges, p1)
            
            # Calculate new quality and update pass counter
            quality1 = self._calculate_quality(edges=edges, partition=p1, m=m)

            # Declare stopping criterion and update old quality
            canPass = (quality1 - quality0 > self.min_quality_gain) and (
                _pass < self.max_pass
            )
            quality0 = quality1

            self.quality_ = quality0

            # Update ret and compress graph (Phase 3: Aggregation)
            if canPass:
                ret = ret.join(
                    other=p1.selectExpr(f"id as pass{_pass}", f"c as pass{_pass + 1}"),
                    on=f"pass{_pass}",
                ).checkpoint()

                # Phase 3: Aggregate network based on the refined partition
                edges = (
                    self._label_edges(edges, refined_p1)
                    .select("cSrc", "cDst", "weight")
                    .groupBy("cSrc", "cDst")
                    .agg(F.sum("weight").alias("weight"))
                    .selectExpr("cSrc as src", "cDst as dst", "weight")
                ).checkpoint()

            prev_p2.unpersist()
            _pass += 1

        # Set modularity (regardless of quality function used)
        self.modularity_ = self._calculate_modularity(edges=edges, partition=p1, m=m)

        # Return final dataframe with sorted columns
        if self.all_partitions:
            # Return sorted columns
            cols = self._sort_passes(ret)
            ret = ret.select(cols)

        # Return final dataframe with id & community
        else:
            _last = self._last_pass(ret)
            ret = ret.selectExpr("id as node_id", f"{_last} as cluster")

        self.labels_ = ret

        return self

    def _verify_data(self, df, expected_cols_grouping, expected_cols_others):
        """Checks if `edges` meets the format expected by `LeidenCommunities`.

        Args:
            df (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.
                It must have `src` and `dst` as its columns. The user may also
                specify the weight of each edge via the additional `weight` column
                (optional).

            expected_cols_grouping (list):
                A list of strings representing the columns that must be present in
                `df` to group the data.

            expected_cols_others (list):
                A list of strings representing the columns that must be present in
                `df` but are not used for grouping.
        """

        cols = df.columns
        expected_cols = expected_cols_grouping + expected_cols_others

        # Check type
        if not isinstance(df, DataFrame):
            raise TypeError("Input data must be a pyspark DataFrame.")

        # Check missing columns
        msg = "Input data is missing expected column '{}'."
        for col in expected_cols:
            if col not in cols:
                raise ValueError(msg.format(col))

        # Check for duplicates
        dup = (
            df.groupBy(*expected_cols_grouping)
            .agg(F.count(F.lit(1)).alias("count"))
            .where("count > 1")
            .count()
        )
        if dup > 0:
            raise ValueError("Data has duplicated entries.")

    def _last_pass(self, df):
        """Returns the column name of the last pass.

        Args:
            df (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the series of partitions made by
                `LeidenCommunities` (a dataframe with columns 'id', 'pass0',
                'pass1', 'pass2', 'pass3', etc.).
        """

        # Get all `passX` columns as list
        cols = [col for col in df.columns if "pass" in col]

        # Get last pass as int
        _max = max([int(col.split("pass")[1]) for col in cols])

        # Return last pass as string
        return f"pass{_max}"

    def _label_degrees(self, edges, partition):
        """
        This function uses the edges of a graph to calculate the weighted degrees
        of each node and joins the result with the partition passed by the user.

        Args:
            edges (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.
                It must have `src` and `dst` as its columns. The user may also
                specify the weight of each edge via the additional `weight` column
                (optional).

            partition (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the partition of an undirected
                graph (i.e., a table that indicates the community that each node
                belongs to). The dataframe must have columns `id` (indicating each
                node's ID) and `c` (indicating each node's assigned community).

        Returns:
            (Dataframe):
                This function returns a dataframe with columns `id` (representing the ID
                of each node in the graph), `c` (representing each node's community) and
                `degree` (representing each node's degree).
        """

        # Get id, community and weighted degree
        ret = (
            partition.join(
                # Unite sources and destinations to avoid double join
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
        """This function uses `partition` to add two columns to `edges`. The added
        columns `cSrc` and `cDst` indicate the community that the source and
        destination nodes belong to.

            Args:
            edges (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.
                It must have `src` and `dst` as its columns. The user may also
                specify the weight of each edge via the additional `weight` column
                (optional).

            partition (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the partition of an undirected
                graph (i.e., a table that indicates the community that each node
                belongs to). The dataframe must have columns `id` (indicating each
                node's ID) and `c` (indicating each node's assigned community).

        Returns:
            (pyspark.sql.dataframe.DataFrame):
                This function returns `edges` with two additional columns: the community
                that the source node belongs to (`cSrc`) and the community that the
                destination node belongs to (`cDst`).
        """

        # Get communities
        ret = (
            edges
            # Start off with src, dst and weight
            .select("src", "dst", "weight")
            # Source destination
            .join(
                other=partition.selectExpr("id as src", "c as cSrc"),
                on="src",
                how="left",
            )
            # Destination community
            .join(
                other=partition.selectExpr("id as dst", "c as cDst"),
                on="dst",
                how="left",
            ).checkpoint()
        )

        return ret

    def _calculate_m(self, edges) -> int:
        """Get the weighted size of an undirected graph (where $m$ is
        defined as $m = \\frac{1}{2} \\sum_{ij} A_{ij}$)).

        Args:
            edges (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.
                It must have `src` and `dst` as its columns. The user may also
                specify the weight of each edge via the additional `weight` column
                (optional).

        Returns:
            (int): Returns the weighted size of the graph.
        """

        m = edges.select(F.sum("weight")).collect()[0][0]

        return int(m)

    def _calculate_quality(self, edges, partition, m=None) -> float:
        """This function calculates the quality of a partition based on the selected
        quality function (modularity or CPM).

        Args:
            edges (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.
                It must have `src` and `dst` as its columns. The user may also
                specify the weight of each edge via the additional `weight` column
                (optional).

            partition (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the partition of an undirected
                graph (i.e., a table that indicates the community that each node
                belongs to). The dataframe must have columns `id` (indicating each
                node's ID) and `c` (indicating each node's assigned community).

            m (int):
                The weighted size of the graph (the output of `_calculate_m()`).

        Returns:
            (float):
                Quality of the partition according to the selected quality function.
        """
        
        if self.quality_function == "modularity":
            return self._calculate_modularity(edges, partition, m)
        else:  # CPM - Constant Potts Model
            return self._calculate_cpm(edges, partition, m)
            
    def _calculate_modularity(self, edges, partition, m=None) -> float:
        """This function calculates the modularity of a partition.

        Args:
            edges (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.
                It must have `src` and `dst` as its columns. The user may also
                specify the weight of each edge via the additional `weight` column
                (optional).

            partition (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the partition of an undirected
                graph (i.e., a table that indicates the community that each node
                belongs to). The dataframe must have columns `id` (indicating each
                node's ID) and `c` (indicating each node's assigned community).

            m (int):
                The weighted size of the graph (the output of `_calculate_m()`).

        Returns:
            (float):
                Bound between -1 and 1 representing the modularity of a
                partition. The output may exceed these bounds depending on the value of
                `resolution` (which is set to 1 by default).
        """

        # Calculate m (if necessary) and norm
        m = self._calculate_m(edges) if m is None else m
        norm = 1 / (2 * m)

        # Declare basic inputs
        labeledEdges = self._label_edges(edges, partition)
        labeledDegrees = self._label_degrees(edges, partition)

        # Get term on LHS
        k_in = (labeledEdges.where("cSrc = cDst").select(F.sum("weight"))).collect()[0][
            0
        ]

        # Handle NoneType
        k_in = 0 if k_in is None else k_in

        # Get term on RHS
        k_out = (
            labeledDegrees.groupby("c")
            .agg(F.sum("degree").alias("kC"))
            .selectExpr(f"{self.resolution} * sum(kC * kC)")
        ).collect()[0][0]

        # Return modularity
        return (k_in / m) - (norm**2 * float(k_out))
        
    def _calculate_cpm(self, edges, partition, m=None) -> float:
        """Calculate the Constant Potts Model (CPM) quality of a partition.
        
        CPM is defined as:
        sum_c [e_c - γ * (n_c choose 2)]
        where:
        - e_c is the number of internal edges of community c
        - n_c is the number of nodes in community c
        - γ is the resolution parameter
        
        Args:
            edges (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.
            
            partition (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the partition of an undirected graph.
                
            m (int):
                The weighted size of the graph (not used in CPM but kept for interface consistency).
                
        Returns:
            (float): CPM quality of the partition
        """
        # Get labeled edges
        labeledEdges = self._label_edges(edges, partition)
        
        # Get internal edges for each community
        internal_edges = (
            labeledEdges.where("cSrc = cDst")
            .groupBy("cSrc")
            .agg(F.sum("weight").alias("internal_edges"))
            .withColumnRenamed("cSrc", "c")
        )
        
        # Get number of nodes in each community
        community_sizes = (
            partition.groupBy("c")
            .count()
            .withColumnRenamed("count", "size")
        )
        
        # Join and calculate CPM
        cpm = (
            internal_edges
            .join(community_sizes, on="c", how="full_outer")
            .fillna(0)
            .withColumn(
                "cpm_term",
                F.col("internal_edges") - 
                self.resolution * (F.col("size") * (F.col("size") - 1)) / 2
            )
            .agg(F.sum("cpm_term"))
            .collect()[0][0]
        )
        
        return float(cpm)

    def _fast_local_move(self, edges, partition, m=None):
        """
        Fast local move phase of the Leiden algorithm. Similar to the Louvain local move,
        but with optimizations to avoid revisiting stable nodes.

        Args:
            edges (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.
                It must have `src` and `dst` as its columns. The user may also
                specify the weight of each edge via the additional `weight` column
                (optional).

            partition (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the partition of an undirected
                graph (i.e., a table that indicates the community that each node
                belongs to). The dataframe must have columns `id` (indicating each
                node's ID) and `c` (indicating each node's assigned community).

            m (int):
                The weighted size of the graph (the output of `_calculate_m()`).

        Returns:
            (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe with the same number of rows as there are vertices.
                Columns `cx` and `cj` represent each node's current and optimal
                neighboring community (accordingly).
        """
        
        # Calculate m if necessary
        m = self._calculate_m(edges) if m is None else m

        # Label edges and degrees here to avoid long lineages
        labeledDegrees = self._label_degrees(edges, partition)
        labeledEdges = self._label_edges(edges, partition)

        # Add sum(ki) for i in C to labeledDegrees
        dq = (
            labeledDegrees.withColumn(
                "cx_sum_ki", F.sum("degree").over(Window.partitionBy("c"))
            )
            # Get sum(Aix) for i in Cx\{x}
            .join(
                other=(
                    labeledEdges.where("(src != dst) and (cSrc = cDst)")
                    .selectExpr("src as id", "weight")
                    .unionByName(
                        labeledEdges.where("(src != dst) and (cSrc = cDst)").selectExpr(
                            "dst as id", "weight"
                        )
                    )
                    .groupBy("id")
                    .agg(F.sum("weight").alias("cx_sum_aix"))
                ),
                on="id",
                how="left",
            )
            # Get sum(Aix) for i in Cj (relationship 1:J)
            .join(
                other=(
                    labeledEdges.where("cSrc != cDst")
                    .selectExpr("src as id", "cDst as cj", "weight")
                    .unionByName(
                        labeledEdges.where("cSrc != cDst").selectExpr(
                            "dst as id", "cSrc as cj", "weight"
                        )
                    )
                    .groupBy("id", "cj")
                    .agg(F.sum("weight").alias("cj_sum_aix"))
                ),
                on="id",
                how="left",
            )
            # Get sum(ki) for i in Cj
            .join(
                other=(
                    labeledDegrees.withColumnRenamed("c", "cj")
                    .groupBy("cj")
                    .agg(F.sum("degree").alias("cj_sum_ki"))
                ),
                on="cj",
                how="left",
            )
        )
        
        # Calculate quality change based on quality function
        if self.quality_function == "modularity":
            dq = dq.withColumn(
                "mdq",
                F.coalesce("cj_sum_aix", F.lit(0))
                - F.coalesce("cx_sum_aix", F.lit(0))
                - (
                    F.col("degree")
                    / F.lit(2 * m)
                    * self.resolution
                    * (F.col("cj_sum_ki") - F.col("cx_sum_ki") + F.col("degree"))
                ),
            )
        else:  # CPM
            dq = dq.withColumn(
                "mdq",
                F.coalesce("cj_sum_aix", F.lit(0))
                - F.coalesce("cx_sum_aix", F.lit(0))
                - (
                    self.resolution * (
                        F.size("cj") - F.size("c") + 1
                    )
                ),
            )
            
        # Rank quality changes and select best move
        dq = (
            dq
            # Rank mdq(x) in descending order
            .select(
                F.col("id"),
                F.col("c"),
                F.coalesce("cj", F.col("c")).alias("cj"),  # Trapped nodes: Cx == Cj
                F.col("mdq"),
                F.row_number()
                .over(Window.partitionBy("id").orderBy(F.desc("mdq")))
                .alias("mdq_rank"),
            )
            # Keep best (or first) change
            .where(F.col("mdq_rank") == 1)
        )

        # Break symmetric swaps
        dq = (
            dq.withColumn(
                "sym_rank",
                F.row_number().over(
                    Window.partitionBy(
                        F.sort_array(F.array(F.col("c"), F.col("cj")))
                    ).orderBy(F.desc("mdq"))
                ),
            )
            # Select best switch (cStar) and break symmetric swaps with sym_rank
            .withColumn(
                "cStar",
                F.when(
                    ((F.col("mdq") > F.lit(1e-04)) & (F.col("sym_rank") == 1)),
                    F.col("cj"),
                ).otherwise(F.col("c")),
            ).selectExpr("id", "c as cx", "cStar as cj")
        )

        return dq

    def _refine_partition(self, edges, partition):
        """
        Refine the partition to ensure that communities are well-connected.
        This is the key innovation of the Leiden algorithm over Louvain.

        Args:
            edges (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the edges of an undirected graph.

            partition (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the partition of an undirected graph.

        Returns:
            (pyspark.sql.dataframe.DataFrame):
                A refined partition with well-connected communities.
        """
        # Label edges with community information
        labeled_edges = self._label_edges(edges, partition)
        
        # Get connected components within communities
        # First, filter to only edges within communities
        internal_edges = labeled_edges.where("cSrc = cDst")
        
        # Use GraphFrames to find connected components within each community
        # We need to create an aggregate version to represent the connected components
        nodes = (
            internal_edges.selectExpr("src as id")
            .unionByName(internal_edges.selectExpr("dst as id"))
            .distinct()
        )
        
        # Find connected components
        from pyspark.sql import SparkSession
        from graphframes import GraphFrame
        
        spark = SparkSession.builder.getOrCreate()
        g = GraphFrame(nodes, internal_edges.select("src", "dst"))
        
        # Use GraphFrames' connected components algorithm
        components = g.connectedComponents()
        
        # Join components with original partition
        refined = (
            partition
            .join(
                components.selectExpr("id", "component"),
                on="id",
                how="left"
            )
            # Create a new community ID using both the original community and component
            .withColumn(
                "refined_c",
                F.when(
                    F.col("component").isNotNull(),
                    F.concat(F.col("c"), F.lit("_"), F.col("component"))
                )
                .otherwise(F.col("c"))
            )
            .select("id", "refined_c", "c")
        )
        
        # Create a map from refined community back to original community for aggregation
        refined_to_original = refined.select("refined_c", "c").distinct()
        
        # Return the refined partition
        return refined.select("id", "refined_c as c")

    def _sort_passes(self, res) -> list:
        """Takes the output of `LeidenCommunities` and returns a list containing
        its columns ordered by their integer part in ascending order.
        For example, if the columns returned by `LeidenCommunities are
        `['pass2', 'id', 'pass1', 'pass0']`, this function will turn the list to
        `['id', 'pass0', 'pass1', 'pass2']`.
        This function also supports cases where `max_pass > 10`.

        Args:
            res (pyspark.sql.dataframe.DataFrame):
                A pyspark dataframe representing the output of `LeidenCommunities`.
                `res` must have columns 'id', 'pass0', 'pass1', 'pass2', etc.
        """

        # Get pass-columns and sort them by their integer part
        cols = [col for col in res.columns if "pass" in col]
        ints = sorted([int(col.replace("pass", "")) for col in cols])
        cols_sorted = ["id"] + ["pass" + str(i) for i in ints]

        return cols_sorted