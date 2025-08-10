import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

class PageRankAnimator:
    def __init__(self, pagerank_obj, top_k=100):
        self.pagerank_obj = pagerank_obj
        self.top_k = top_k
        self.iteration_ranks = []
        self.top_nodes = []
        self.graph = None

    def compute_with_history(self):
        """Run PageRank and store ranks after each iteration."""
        self.pagerank_obj.load_graph()

        N = self.pagerank_obj.N
        M = self.pagerank_obj.M
        d = self.pagerank_obj.d
        teleport = (1.0 - d) / N
        ranks = np.full(N, 1.0 / N)

        for _ in range(self.pagerank_obj.max_iterations):
            new_ranks = teleport + d * M.dot(ranks)
            self.iteration_ranks.append(new_ranks.copy())
            if np.abs(new_ranks - ranks).sum() < self.pagerank_obj.min_diff:
                break
            ranks = new_ranks

        # Select top-K from final iteration
        final_ranks = self.iteration_ranks[-1]
        sorted_indices = np.argsort(final_ranks)[::-1][:self.top_k]
        self.top_nodes = [node for node, idx in self.pagerank_obj.node_to_index.items() if idx in sorted_indices]

        # Build subgraph
        self._build_subgraph(sorted_indices)

    def _build_subgraph(self, top_indices):
        """Extract subgraph for top-K nodes."""
        index_to_node = {idx: node for node, idx in self.pagerank_obj.node_to_index.items()}
        self.graph = nx.DiGraph()
        for idx in top_indices:
            self.graph.add_node(index_to_node[idx])
        M = self.pagerank_obj.M
        for i in top_indices:
            for j in top_indices:
                if M[i, j] != 0:
                    self.graph.add_edge(index_to_node[j], index_to_node[i])  # j->i

    def animate(self):
        """Animate PageRank evolution."""
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42)

        def update(frame):
            ax.clear()
            ranks = self.iteration_ranks[frame]
            node_sizes = [
                ranks[self.pagerank_obj.node_to_index[node]] * 5000
                for node in self.graph.nodes()
            ]
            nx.draw_networkx(self.graph, pos, ax=ax, with_labels=True,
                             node_size=node_sizes, node_color="skyblue",
                             edge_color="gray", font_size=8)
            ax.set_title(f"PageRank Iteration {frame + 1}")

        anim = FuncAnimation(fig, update, frames=len(self.iteration_ranks), interval=500, repeat=False)
        plt.show()
