import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math
from matplotlib.patches import FancyArrowPatch

class PageRankAnimator:
    def __init__(self, pagerank_obj, top_k=50):
        self.pagerank_obj = pagerank_obj
        self.top_k = top_k
        self.iteration_ranks = []
        # colormap
        self.blue_colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', 
                           '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1']
        self.last_frame_data = None

    def compute_with_history(self):
        """Runs PageRank and stores rank history for animation"""
        self.pagerank_obj.load_graph()
        N = self.pagerank_obj.N
        M = self.pagerank_obj.M
        d = self.pagerank_obj.d
        teleport = (1.0 - d) / N
        ranks = np.full(N, 1.0 / N)

        self.iteration_ranks.append(ranks.copy())

        for _ in range(self.pagerank_obj.max_iterations):
            new_ranks = teleport + d * M.dot(ranks)
            self.iteration_ranks.append(new_ranks.copy())
            if np.abs(new_ranks - ranks).sum() < self.pagerank_obj.min_diff:
                break
            ranks = new_ranks

    def _get_top_nodes_for_iteration(self, iteration_ranks):
        """Get top K nodes (id and rank) for a specific iteration"""
        sorted_indices = np.argsort(iteration_ranks)[::-1][:self.top_k]
        top_nodes = []
        # use reverse map for faster lookups
        index_to_node = getattr(self.pagerank_obj, "index_to_node", None)
        for idx in sorted_indices:
            node_id = index_to_node[idx] if index_to_node else next(
                node for node, i in self.pagerank_obj.node_to_index.items() if i == idx
            )
            top_nodes.append((node_id, iteration_ranks[idx]))
        return top_nodes

    def _create_hierarchical_positions(self, num_nodes):
        """Creates a grid-like tree structure for node positions with increased width for levels 3 and 4"""
        positions = {}
        # Modified level structure with increased width for levels 3 and 4
        level_structure = [(1, 1), (1, 3), (2, 6), (3, 8), (4, 10)]  
        node_idx = 0
        y_offset = 0
        
        for level, (rows, cols) in enumerate(level_structure):
            if node_idx >= num_nodes:
                break
            max_nodes_in_level = rows * cols
            nodes_in_this_level = min(max_nodes_in_level, num_nodes - node_idx)
            level_positions = []
            nodes_placed = 0
            
            for row in range(rows):
                if nodes_placed >= nodes_in_this_level:
                    break
                nodes_in_row = min(cols, nodes_in_this_level - nodes_placed)
                if nodes_in_row == 1:
                    x_positions = [0]
                else:
                    # Increased spacing for levels 3 and 4
                    spacing = 3.5 if level >= 2 else 2.5
                    x_span = (nodes_in_row - 1) * spacing
                    x_positions = np.linspace(-x_span/2, x_span/2, nodes_in_row)
                y_pos = y_offset - row * 2.5
                for x_pos in x_positions:
                    if nodes_placed < nodes_in_this_level:
                        level_positions.append((x_pos, y_pos))
                        nodes_placed += 1
            
            for i, pos in enumerate(level_positions):
                if node_idx + i < num_nodes:
                    positions[node_idx + i] = pos
            node_idx += nodes_in_this_level
            y_offset -= (rows + 1) * 2.5
        
        if node_idx < num_nodes:
            remaining = num_nodes - node_idx
            cols_final = min(10, remaining) 
            for i in range(remaining):
                row = i // cols_final
                col = i % cols_final
                x_span = (cols_final - 1) * 3.0
                x_pos = -x_span/2 + col * 3.0 if cols_final > 1 else 0
                y_pos = y_offset - row * 2.5
                positions[node_idx + i] = (x_pos, y_pos)
        
        return positions

    def _get_node_level(self, node_index, num_nodes):
        """Determine which hierarchical level a node belongs to"""
        level_structure = [(1, 1), (1, 3), (2, 6), (3, 8), (4, 10)]
        cumulative_nodes = 0
        
        for level, (rows, cols) in enumerate(level_structure):
            max_nodes_in_level = rows * cols
            nodes_in_this_level = min(max_nodes_in_level, num_nodes - cumulative_nodes)
            if node_index < cumulative_nodes + nodes_in_this_level:
                return level
            cumulative_nodes += nodes_in_this_level
        
        return len(level_structure)  

    def _draw_nodes(self, ax, top_nodes_data, positions, max_rank, min_rank):
        """Draws nodes with hierarchical blue coloring and modified text positioning"""
        for i, (node, rank) in enumerate(top_nodes_data):
            if i in positions:
                x, y = positions[i]
                
                # Get level for color assignment
                level = self._get_node_level(i, len(top_nodes_data))
                color_idx = min(level, len(self.blue_colors) - 1)
                node_color = self.blue_colors[color_idx]
                
                # Size based on PageRank score
                norm_rank = (rank - min_rank) / (max_rank - min_rank) if max_rank > min_rank else 0.5
                node_size = 300 + (1200 - 300) * norm_rank
                
                # Draw the node
                ax.scatter(x, y, s=node_size, c=node_color, 
                           alpha=0.8, edgecolors='black', linewidths=2, zorder=2)
                
                # Text above the node (Node ID and PageRank Score)
                scaled_rank = rank * 1000
                above_text = f"ID: {node}\nPR: {scaled_rank:.3f}"
                ax.text(x, y + 0.8, above_text,
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold',
                        color='black', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Rank position on the left side of the node
                rank_text = f"#{i+1}"
                ax.text(x - 1.2, y, rank_text,
                        ha='right', va='center',
                        fontsize=10, fontweight='bold',
                        color='darkblue',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))


    def _draw_arrow(self, ax, x1, y1, x2, y2, **kwargs):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>', mutation_scale=10,
                                lw=1.5, **kwargs)
        ax.add_patch(arrow)

    def _draw_top10_links_grid(self, ranks_vector, k_top=10, k_in=5, k_out=5, show=False, filename=None):
        """Create a 2x5 grid. In each cell: center is top node, left = incoming (<=5), right = outgoing (<=5)."""
        pr = self.pagerank_obj
        top_idxs = pr.get_top_k_indices(ranks_vector, k_top)

        fig, axes = plt.subplots(2, 5, figsize=(28, 10))
        axes = axes.flatten()

        for i, node_idx in enumerate(top_idxs):
            ax = axes[i]
            ax.set_title(f"Top #{i+1}  (ID {pr.index_to_node[node_idx]})", fontsize=10, pad=6)

            # Layout coords
            cx, cy = 0.0, 0.0
            in_x = -1.8
            out_x = 1.8
            # Vertical spacing
            def y_positions(n):
                if n == 0:
                    return []
                if n == 1:
                    return [0.0]
                # center-around 0
                return np.linspace(-1.2, 1.2, n)

            ins, outs = pr.get_neighbors_limited(node_idx, k_out=k_out, k_in=k_in)

            # center node
            center_pr = ranks_vector[node_idx] * 1000.0
            ax.scatter([cx], [cy], s=800, c="#FFD54F", edgecolors='black', zorder=3)
            ax.text(cx, cy+0.15, f"PR:{center_pr:.2f}", ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
            ax.text(cx, cy-0.22, f"ID:{pr.index_to_node[node_idx]}", ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))

            # incoming (left)
            y_in = y_positions(len(ins))
            for yi, nidx in zip(y_in, ins):
                ax.scatter([in_x], [yi], s=220, c="#A5D6A7", edgecolors='black', zorder=2)
                ax.text(in_x, yi+0.12, f"{pr.index_to_node[nidx]}", ha='center', va='bottom', fontsize=8)
                # arrow from neighbor -> center
                self._draw_arrow(ax, in_x+0.15, yi, cx-0.2, cy, color="#2E7D32")

            # outgoing (right)
            y_out = y_positions(len(outs))
            for yo, nidx in zip(y_out, outs):
                ax.scatter([out_x], [yo], s=220, c="#90CAF9", edgecolors='black', zorder=2)
                ax.text(out_x, yo+0.12, f"{pr.index_to_node[nidx]}", ha='center', va='bottom', fontsize=8)
                # arrow from center -> neighbor
                self._draw_arrow(ax, cx+0.2, cy, out_x-0.15, yo, color="#1565C0")

            # cosmetics
            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-1.8, 1.8)
            ax.axis('off')

        # Title for the incoming and outgoing links
        fig.suptitle("Top 10 Nodes with up to 5 Incoming (left, green) and 5 Outgoing (right, blue) Links",
                     fontsize=14, fontweight='bold', y=0.98)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ----------------- Animation & Saving -----------------

    def animate(self):
        """Animate the top K nodes for each iteration and, on the last frame, show the top-10 links view."""
        fig, ax = plt.subplots(figsize=(24, 16))  
        all_ranks = np.concatenate(self.iteration_ranks)
        max_rank = np.max(all_ranks)
        min_rank = np.min(all_ranks)
        self.last_frame_data = None

        total_frames = len(self.iteration_ranks)

        def update(frame):
            ax.clear()
            current_ranks = self.iteration_ranks[frame]
            top_nodes_data = self._get_top_nodes_for_iteration(current_ranks)
            if frame == total_frames - 1:
                self.last_frame_data = {
                    'top_nodes_data': top_nodes_data,
                    'frame': frame,
                    'ranks_vector': current_ranks
                }
            positions = self._create_hierarchical_positions(len(top_nodes_data))
            self._draw_nodes(ax, top_nodes_data, positions, max_rank, min_rank)

            ax.set_title(f'Top {self.top_k} PageRank Nodes - Iteration {frame + 1}', 
                        fontsize=18, fontweight='bold', pad=20)
            
            #legend (top left)
            legend_text = ("Blue Gradient: Lighter = Higher Level, Darker = Lower Level\n"
                          "Node Size: Proportional to PageRank Score\n"
                          "Yellow boxes show ranking position")
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            #iteration and PageRank statistics (bottom right)
            max_pr = np.max(current_ranks) * 1000  # Scale to match display
            min_pr = np.min(current_ranks) * 1000  # Scale to match display
            total_nodes = len(current_ranks)
            stats_text = (f"Iteration: {frame + 1}\n"
                         f"Total Nodes: {total_nodes}\n"
                         f"Max PageRank: {max_pr:.4f}\n"
                         f"Min PageRank: {min_pr:.4f}")
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            
            if positions:
                all_x = [pos[0] for pos in positions.values()]
                all_y = [pos[1] for pos in positions.values()]
                ax.set_xlim(min(all_x) - 4, max(all_x) + 4)
                ax.set_ylim(min(all_y) - 4, max(all_y) + 4)
            ax.axis('off')

            if frame == total_frames - 1:
                self._draw_top10_links_grid(
                    self.last_frame_data['ranks_vector'],
                    k_top=10, k_in=5, k_out=5,
                    show=True,
                    filename='top10_links_5in5out.png'
                )

        anim = FuncAnimation(fig, update, frames=total_frames, interval=1200, repeat=False)
        plt.tight_layout()
        plt.show()
        return anim

    def save_final_frame(self, filename='final_pagerank_frame.png'):
        """Save the final frame as a static image (Top 25 hierarchical view)."""
        if not self.last_frame_data:
            print("No final frame data available. Run animation first.")
            return
        
        fig, ax = plt.subplots(figsize=(24, 16))
        top_nodes_data = self.last_frame_data['top_nodes_data']
        all_ranks = np.concatenate(self.iteration_ranks)
        max_rank = np.max(all_ranks)
        min_rank = np.min(all_ranks)
        positions = self._create_hierarchical_positions(len(top_nodes_data))
        
        # Draw final nodes
        self._draw_nodes(ax, top_nodes_data, positions, max_rank, min_rank)

        ax.set_title(f'Final PageRank Results - Top {self.top_k} Nodes', 
                    fontsize=18, fontweight='bold', pad=20)
        
        #legend (top left)
        legend_text = ("Blue Gradient: Lighter = Higher Level, Darker = Lower Level\n"
                      "Node Size: Proportional to PageRank Score\n"
                      "Yellow boxes show ranking position")
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # Add final statistics (bottom right)
        final_ranks = self.iteration_ranks[-1]
        max_pr = np.max(final_ranks) * 1000  # Scale to match display
        min_pr = np.min(final_ranks) * 1000  # Scale to match display
        total_nodes = len(final_ranks)
        final_iteration = len(self.iteration_ranks)
        stats_text = (f"Final Iteration: {final_iteration}\n"
                     f"Total Nodes: {total_nodes}\n"
                     f"Max PageRank: {max_pr:.4f}\n"
                     f"Min PageRank: {min_pr:.4f}")
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        if positions:
            all_x = [pos[0] for pos in positions.values()]
            all_y = [pos[1] for pos in positions.values()]
            ax.set_xlim(min(all_x) - 4, max(all_x) + 4)
            ax.set_ylim(min(all_y) - 4, max(all_y) + 4)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Final frame saved as: {filename}")
        plt.close()

    # Methd to save image for top 10 nodes with incoming and outgoing links
    def save_top10_links_figure(self, filename='top10_links_5in5out.png'):
        if not self.last_frame_data:
            print("No final frame ranks available. Run animation first.")
            return
        ranks = self.last_frame_data['ranks_vector']
        self._draw_top10_links_grid(ranks, k_top=10, k_in=5, k_out=5, show=False, filename=filename)
