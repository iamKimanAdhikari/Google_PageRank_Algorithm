import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix

class PageRank:
    def __init__(self, max_edges=None, max_iterations=100, min_diff=1e-6, d=0.85, cache_file="pagerank_cache.pkl"):
        # Paths and parameters
        self.filepath = Path(__file__).parent.parent / "Dataset"
        self.file = self.filepath / "web-Google.txt"
        self.max_edges = max_edges
        self.max_iterations = max_iterations
        self.min_diff = min_diff
        self.d = d
        self.cache_file = Path(cache_file)

    def load_graph(self):
        """Load graph into sparse matrix"""

        # Load from cache
        if self.cache_file.exists():
            print("Loading cached graph...")
            with open(self.cache_file, "rb") as f:
                self.M, self.node_to_index, self.N = pickle.load(f)
            return

        # Read dataset
        print("Reading dataset...")
        df = pd.read_csv(
            self.file,
            sep="\t",
            comment="#",
            names=["From", "To"],
            nrows=self.max_edges
        )

        # Create node map
        nodes = pd.concat([df["From"], df["To"]]).unique()
        self.node_to_index = {node: i for i, node in enumerate(nodes)}
        self.N = len(nodes)

        # Convert to indices
        row_idx = df["From"].map(self.node_to_index)
        col_idx = df["To"].map(self.node_to_index)

        # Calculate out-degree
        out_degree = df.groupby("From").size()
        out_degree_map = out_degree.to_dict()

        # Assign weights
        weights = [1.0 / out_degree_map[src] for src in df["From"]]

        # Build sparse matrix
        self.M = csr_matrix((weights, (col_idx, row_idx)), shape=(self.N, self.N))

        # Save to cache
        print("Saving graph to cache...")
        with open(self.cache_file, "wb") as f:
            pickle.dump((self.M, self.node_to_index, self.N), f)

    def calculate_pagerank(self):
        """Run iterative PageRank"""
        ranks = np.full(self.N, 1.0 / self.N)
        teleport = (1.0 - self.d) / self.N

        for iteration in range(self.max_iterations):
            # PR formula
            new_ranks = teleport + self.d * self.M.dot(ranks)

         

        # Return {node: score}
        return {node: ranks[idx] for node, idx in self.node_to_index.items()}
