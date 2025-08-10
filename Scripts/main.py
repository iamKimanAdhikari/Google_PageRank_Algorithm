from pagerank import PageRank
from animate_pagerank import PageRankAnimator

def main():
    pr = PageRank(max_edges=10000, max_iterations=50)
    animator = PageRankAnimator(pr, top_k=100)
    animator.compute_with_history()
    animator.animate()

if __name__ == "__main__":
    main()
