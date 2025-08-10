import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix

class PageRank:
    def __init__(self, max_edges=None, max_iterations=100, min_diff=1e-6, d=0.85, cache_file="pagerank_cache.pkl"):
        # Base parameters
        self.filepath = Path(__file__).parent.parent / "Dataset"
        self.file = self.filepath / "web-Google.txt"
        self.max_edges = max_edges
        self.max_iterations = max_iterations
        self.min_diff = min_diff
        self.d = d
        self.cache_file = Path(cache_file)

    def load_graph(self):
        """Load and preprocess the graph into a sparse transition matrix (with caching)."""

        # If cache exists, load directly to save time
        if self.cache_file.exists():
            print("Loading cached graph data...")
            with open(self.cache_file, "rb") as f:
                self.M, self.node_to_index, self.N = pickle.load(f)
            return

        print("Reading dataset...")
        df = pd.read_csv(
            self.file,
            sep="\t",
            comment="#",
            names=["From", "To"],
            nrows=self.max_edges
        )

        # Build node list and mapping (concat replaces old .append method)
        nodes = pd.concat([df["From"], df["To"]]).unique()
        self.node_to_index = {node: i for i, node in enumerate(nodes)}
        self.N = len(nodes)

        # Map nodes to integer indices for matrix construction
        row_idx = df["From"].map(self.node_to_index)
        col_idx = df["To"].map(self.node_to_index)

        # Calculate out-degree for each node (needed for probability weights)
        out_degree = df.groupby("From").size()
        out_degree_map = out_degree.to_dict()

        # Each edge gets weight = 1 / (out-degree of source)
        weights = [1.0 / out_degree_map[src] for src in df["From"]]

        # Build sparse column-stochastic matrix (CSR format for speed)
        # NOTE: Transposed logic → M[j, i] = probability from i → j
        self.M = csr_matrix((weights, (col_idx, row_idx)), shape=(self.N, self.N))

        # Save to cache for future runs
        print("Saving graph to cache...")
        with open(self.cache_file, "wb") as f:
            pickle.dump((self.M, self.node_to_index, self.N), f)

    def calculate_pagerank(self):
        """Run the PageRank iterative calculation."""
        ranks = np.full(self.N, 1.0 / self.N)  # Start with equal probability for all nodes
        teleport = (1.0 - self.d) / self.N     # Constant teleportation value

        for iteration in range(self.max_iterations):
            # Core PageRank formula: PR = teleport + d * (M * PR)
            new_ranks = teleport + self.d * self.M.dot(ranks)

            # Check convergence (L1 norm difference)
            diff = np.abs(new_ranks - ranks).sum()
            print(f"Iteration {iteration+1}: diff={diff:.6e}")

            ranks = new_ranks
            if diff < self.min_diff:
                print(f"Converged after {iteration+1} iterations.")
                break

        # Return as dictionary {node_id: rank_score}
        return {node: ranks[idx] for node, idx in self.node_to_index.items()}

def main():
    pr = PageRank(
        max_edges=None,         # entire dataset
        max_iterations=100,     # Limit iterations
        min_diff=1e-6,           # Convergence tolerance
        d=0.85,                  # Damping factor
        cache_file="pagerank_cache.pkl"
    )
    pr.load_graph()
    ranks = pr.calculate_pagerank()

    # Show top 10 ranked nodes
    top_100 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:100]
    print("\nTop 100 nodes by PageRank score:")
    for node, score in top_100:
        print(f"Node {node}: {score:.6f}")

if __name__ == "__main__":
    main()
