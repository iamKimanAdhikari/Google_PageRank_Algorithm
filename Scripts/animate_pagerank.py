import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math

class PageRankAnimator:
    def __init__(self, pagerank_obj, top_k=50):
        self.pagerank_obj = pagerank_obj
        self.top_k = top_k
        self.iteration_ranks = []
        # colormap
        self.blue_colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', 
                           '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1']

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
        """Get top K nodes for a specific iteration"""
        sorted_indices = np.argsort(iteration_ranks)[::-1][:self.top_k]
        top_nodes = []
        for idx in sorted_indices:
            for node, node_idx in self.pagerank_obj.node_to_index.items():
                if node_idx == idx:
                    top_nodes.append((node, iteration_ranks[idx]))
                    break
        return top_nodes

    def _create_hierarchical_positions(self, num_nodes):
        """Creates a grid-like tree structure for node positions with increased width for levels 3 and 4"""
        positions = {}
        # Modified level structure with increased width for levels 3 and 4
        level_structure = [(1, 1), (1, 3), (2, 6), (3, 8), (4, 10)]  # Increased cols for levels 3&4
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
            cols_final = min(10, remaining)  # Increased final level width
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
        
        return len(level_structure)  # Final level for remaining nodes

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

    def animate(self):
        """Animate the top K nodes for each iteration"""
        fig, ax = plt.subplots(figsize=(24, 16))  # Increased figure size for wider layout
        all_ranks = np.concatenate(self.iteration_ranks)
        max_rank = np.max(all_ranks)
        min_rank = np.min(all_ranks)
        self.last_frame_data = None

        def update(frame):
            ax.clear()
            current_ranks = self.iteration_ranks[frame]
            top_nodes_data = self._get_top_nodes_for_iteration(current_ranks)
            if frame == len(self.iteration_ranks) - 1:
                self.last_frame_data = {
                    'top_nodes_data': top_nodes_data,
                    'frame': frame
                }
            positions = self._create_hierarchical_positions(len(top_nodes_data))
            self._draw_nodes(ax, top_nodes_data, positions, max_rank, min_rank)

            ax.set_title(f'Top {self.top_k} PageRank Nodes - Iteration {frame + 1}', 
                        fontsize=18, fontweight='bold', pad=20)
            
            # Add legend (top left)
            legend_text = ("Blue Gradient: Lighter = Higher Level, Darker = Lower Level\n"
                          "Node Size: Proportional to PageRank Score\n"
                          "Yellow boxes show ranking position")
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            # Add iteration and PageRank statistics (bottom right)
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

        anim = FuncAnimation(fig, update, frames=len(self.iteration_ranks), interval=1200, repeat=True)
        plt.tight_layout()
        plt.show()
        return anim

    def save_final_frame(self, filename='final_pagerank_frame.png'):
        """Save the final frame as a static image"""
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
        
        # Add legend (top left)
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