# 📊 PageRank Visualization & Animation

## 📖 Overview

This project implements a **PageRank algorithm with animated visualization**.  
It computes PageRank scores for nodes in a graph and visualizes the **top-k ranked nodes** across iterations using **Matplotlib animations** in a hierarchical layout.

Key features:

- Iterative PageRank calculation with convergence check
- Animated evolution of node rankings
- Hierarchical node positioning with blue gradient coloring
- Node size proportional to PageRank score
- Rank position shown in yellow labels
- Export final visualization as PNG

---

## ⚙️ Features

- ✅ PageRank with iteration history tracking
- ✅ Animated visualization using `matplotlib.animation.FuncAnimation`
- ✅ Hierarchical grid/tree node positioning
- ✅ Customizable `top_k` ranking visualization
- ✅ Save final results as static image

---

## 🛠️ Installation & Setup

### Requirements

Install the required libraries:

```bash
pip install matplotlib numpy
```
