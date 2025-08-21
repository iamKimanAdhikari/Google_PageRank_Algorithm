# 📊 PageRank Visualization & Animation

## 📖 Overview
This project implements the **PageRank algorithm with animated visualization**.  
It computes PageRank scores for nodes in a graph and visualizes the **top-K ranked nodes** across iterations using **Matplotlib animations** in a hierarchical layout.

**Highlights**
- Iterative PageRank with convergence check  
- Animated evolution of node rankings  
- Hierarchical node positioning with blue gradient coloring  
- Node size ∝ PageRank score, rank labels in yellow  
- Exports final visualizations as **SVG** (scalable, crisp)  
- For the **Top-10 nodes**, also saves a grid figure showing up to **5 incoming** and **5 outgoing** links for each top node  

---

## 📦 Dataset

Default input: `Dataset/web-Google.txt` (Google Webgraph, 2002)  
📎 [Dataset on Kaggle](https://www.kaggle.com/datasets/pappukrjha/google-web-graph/data)

Header (tab-separated):
```
# Directed graph (each unordered pair of nodes is saved once): web-Google.txt
# Webgraph from the Google programming contest, 2002
# Nodes: 875713 Edges: 5105039
# FromNodeId    ToNodeId
0   11342
0   824020
0   867923
0   891835
11342   0
11342   27469
...
```

**Format requirement:** tab-separated `FromNodeId <TAB> ToNodeId`, with lines starting `#` treated as comments.  
You can replace this file with your own directed edge list in the same format.

⚠️ **Performance note:** Running PageRank on the full dataset (5M edges, ~875K nodes) may take several minutes and require >4 GB RAM. For testing, use `max_edges` to limit input.

---

## 🗂️ Project Structure
```
├─ Dataset/
│ └─ web-Google.txt           # Graph dataset (edge list)
│
├─ images/                    # Auto-saved output visualizations
│ ├─ final_pagerank_results.svg
│ └─ top10_links_5in5out.svg
│
├─ Scripts/
│ ├─ animate_pagerank.py      # Animation & visualization logic
│ ├─ pagerank.py              # PageRank algorithm (sparse matrix)
│ ├─ main.py                  # Entry point to run PageRank + animation
│ └─ pagerank_cache.pkl       # Cache file (auto-generated)
│
├─ .gitignore
└─ README.md                  # Project documentation
```

---

## 🛠️ Installation
Use Python 3.9+ (recommended). Install dependencies:
```bash
pip install matplotlib numpy pandas scipy
```

If you plan to run on large graphs, consider a virtual environment and ensure you have enough RAM.

---

## ▶️ Quick Start
From the project root:
```bash
python main.py
```
What happens:
- ✅ The graph is loaded (and cached), and PageRank is computed with history  
- ✅ An animation window shows the **Top-25** nodes across iterations in a hierarchical layout  
- ✅ Two SVG image files are saved automatically in the **`images/` folder**:
  - `final_pagerank_results.svg` → Final hierarchical view of **Top-25** nodes  
  - `top10_links_5in5out.svg` → **Top-10** nodes; each sub-plot shows up to **5 incoming** (green) and **5 outgoing** (blue) neighbors  

> **Note:** The Top-10 links grid appears *only on the last animation frame* and is then saved automatically.  
> PageRank values shown on nodes are scaled ×1000 for readability.

### ⚡ Quick Demo (small subset)
For faster runs, edit `main.py`:
```python
from pagerank import PageRank
from animate_pagerank import PageRankAnimator

def main():
    pr = PageRank(max_edges=100000, max_iterations=100)  # limit edges for speed
    animator = PageRankAnimator(pr, top_k=25)
    animator.compute_with_history()
    anim = animator.animate()
    animator.save_final_frame('final_pagerank_results.svg')
    # top10_links_5in5out.svg is auto-saved after last frame
```

---

## 🧩 Fine-Tuning & Customization

### PageRank hyperparameters (`pagerank.py`)
- `d=0.85` — damping factor (teleportation prob). Typical: 0.85–0.9  
- `max_iterations=100` — safety cap on iterations  
- `min_diff=1e-6` — convergence threshold (L1 delta)  
- `max_edges=None` — `None` = full file; set integer to limit edges  

### Visualization controls (`animate_pagerank.py`)
- `PageRankAnimator(..., top_k=25)` — nodes drawn per frame  
- `interval=1200` (ms) in `FuncAnimation` — animation speed  
- Hierarchical layout, color ramp, node size & labels — edit `_create_hierarchical_positions` and `_draw_nodes`  

### Saving images
- Final hierarchical: `animator.save_final_frame('final_pagerank_results.svg')`  
- Top-10 grid: `top10_links_5in5out.svg` (auto-saved at last frame)  
  - You can also trigger explicitly via `animator.save_top10_links_figure()`  

---

## 📸 Example Outputs

> Exact node IDs & scores depend on dataset and convergence, but layout & styling match these.

### 1) Final hierarchical results
![Final PageRank Results](images/final_pagerank_results.svg)  
- Nodes sized by PageRank  
- Blue gradient by hierarchy level  
- Yellow pill = rank (#1, #2, …)  
- Labels show `ID` + `PR×1000`  

### 2) Top-10 nodes with 5 in/out links
![Top 10 Nodes with 5 in/out links](images/top10_links_5in5out.svg)  
- Center (gold) = top node, with ID & PR score  
- Left (green) = up to 5 incoming neighbors (arrows → center)  
- Right (blue) = up to 5 outgoing neighbors (arrows center → neighbor)  

---

## 🧪 Reproducibility Tips
- Limit edges (`max_edges`) for consistency on small samples  
- Algorithm is deterministic given same input and params  
- Don’t modify dataset between runs  

---

## 🩺 Troubleshooting
- **Animation loops/closes too fast** → set `repeat=False` and adjust `interval`  
- **No images saved** → call `save_final_frame(...)`; ensure latest `animate_pagerank.py`  
- **Memory issues** → use `max_edges`, 64-bit Python, close other apps  
- **Top-10 grid missing** → ensure using version with `_draw_top10_links_grid`  

---

## 📈 Performance Hints
- Uses CSR sparse matrix (`scipy.sparse.csr_matrix`) for efficiency  
- Loosen `min_diff` (e.g., `5e-6`) for speed at cost of accuracy  

---

## 🧭 FAQ
- **Different K values?** → change `top_k` in `PageRankAnimator`  
- **Different dataset?** → drop your own TSV (Tab Separated Values) edge list in `Dataset/`  
- **Why scores ×1000?** → only for readability; doesn’t affect math  

---

## ✅ Minimal Example
```python
# main.py
from pagerank import PageRank
from animate_pagerank import PageRankAnimator

def main():
    pr = PageRank(max_edges=100000, max_iterations=100, d=0.85, min_diff=1e-6)
    animator = PageRankAnimator(pr, top_k=25)
    animator.compute_with_history()
    animator.animate()  # displays animation; auto-saves Top-10 grid on last frame
    animator.save_final_frame('final_pagerank_results.svg')  # saves Top-25 final view

if __name__ == "__main__":
    main()
```