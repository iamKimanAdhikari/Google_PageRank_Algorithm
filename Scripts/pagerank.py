import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from collections import defaultdict

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

        # Will be set in load_graph()
        self.M = None
        self.node_to_index = None
        self.index_to_node = None
        self.N = None

        # Adjacency
        self.out_neighbors = None  # dict: idx -> list of neighbor idx
        self.in_neighbors = None   # dict: idx -> list of neighbor idx

    def load_graph(self):
        """Load graph into sparse matrix and build adjacency lists."""

        # Load from cache
        if self.cache_file.exists():
            print("Loading cached graph...")
            with open(self.cache_file, "rb") as f:
                self.M, self.node_to_index, self.N = pickle.load(f)
            print("Rebuilding adjacency from dataset for neighbor lookups...")
            df = pd.read_csv(
                self.file,
                sep="\t",
                comment="#",
                names=["From", "To"],
                nrows=self.max_edges
            )
            nodes = pd.concat([df["From"], df["To"]]).unique()
            self.index_to_node = {i: node for node, i in self.node_to_index.items()}
            self._build_adjacency(df)
            return

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
        self.index_to_node = {i: node for node, i in self.node_to_index.items()}
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

        # Build adjacency lists
        self._build_adjacency(df)

    def _build_adjacency(self, df: pd.DataFrame):
        """Build outgoing and incoming adjacency lists as index->list[index]."""
        out_adj = defaultdict(list)
        in_adj = defaultdict(list)

        for fr, to in zip(df["From"], df["To"]):
            u = self.node_to_index.get(fr)
            v = self.node_to_index.get(to)
            if u is None or v is None:
                continue
            out_adj[u].append(v)
            in_adj[v].append(u)

        # Removing duplicates
        self.out_neighbors = {u: list(dict.fromkeys(vs)) for u, vs in out_adj.items()}
        self.in_neighbors = {v: list(dict.fromkeys(vs)) for v, vs in in_adj.items()}

    def calculate_pagerank(self):
        """Run iterative PageRank and return {node_id: score}."""
        ranks = np.full(self.N, 1.0 / self.N)
        teleport = (1.0 - self.d) / self.N

        for iteration in range(self.max_iterations):
            # PR formula
            new_ranks = teleport + self.d * self.M.dot(ranks)

            # Convergence check
            diff = np.abs(new_ranks - ranks).sum()
            print(f"Iteration {iteration+1}: diff={diff:.6e}")

            ranks = new_ranks
            if diff < self.min_diff:
                print(f"Converged after {iteration+1} iterations.")
                break

        # Return {node: score}
        return {node: ranks[idx] for node, idx in self.node_to_index.items()}

    # Helper methods used by the animator 
    def get_top_k_indices(self, ranks_vector: np.ndarray, k: int):
        """Return the indices of the top-k nodes by rank."""
        k = min(k, len(ranks_vector))
        return np.argsort(ranks_vector)[::-1][:k]

    def get_neighbors_limited(self, node_idx: int, k_out: int = 5, k_in: int = 5):
        """Return up to k_out outgoing and k_in incoming neighbor indices for a node."""
        outs = self.out_neighbors.get(node_idx, [])[:k_out]
        ins = self.in_neighbors.get(node_idx, [])[:k_in]
        return ins, outs
