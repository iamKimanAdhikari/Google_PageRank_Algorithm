from pagerank import PageRank
from animate_pagerank import PageRankAnimator

def main():
    # Initialize PageRank
    pr = PageRank(max_iterations=100)
    
    # Create animator with top k nodes
    animator = PageRankAnimator(pr, top_k=25)
    
    print("Computing PageRank with iteration history...")
    animator.compute_with_history()
    
    print("Starting animation...")
    print("PageRank values are scaled by a factor of 1000 for easier reading")
    
    # Start the animation
    anim = animator.animate()
    
    # Save the final frame after animation
    print("\nSaving final frame...")
    animator.save_final_frame('final_pagerank_results.png')
    

if __name__ == "__main__":
    main()