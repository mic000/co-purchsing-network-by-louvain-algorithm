## Overview
This project studies and implements the two distributed graph clustering algorithms proposed by Hamann et al. (2018):

- **DSLM-Mod** — optimizes **modularity** using a distributed extension of the Louvain algorithm
- **DSLM-Map** — optimizes the **map equation** (minimum description length) using a distributed Infomap-style approach

Since direct access to a Thrill compute cluster is not available, we implement **single-machine analogues** using `python-louvain` and `Infomap`, and evaluate them on the **Amazon co-purchase graph** (com-amazon, SNAP).

---

## Requirements

Python 3.8+ is recommended.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```
networkx==3.2.1
python-louvain==0.16
infomap==2.6.1
matplotlib==3.8.2
```

---

## Dataset

This project uses the **Amazon co-purchase graph** from the [SNAP Large Network Dataset Collection](https://snap.stanford.edu/data/com-Amazon.html).

**Download steps:**

1. Go to https://snap.stanford.edu/data/com-Amazon.html
2. Download `com-amazon.ungraph.txt.gz`
3. Unzip and place `com-amazon.ungraph.txt` in the **project root directory**

> The dataset is ~25 MB compressed (~50 MB unzipped) and is excluded from this repository via `.gitignore`.

| Property | Value |
|---|---|
| Nodes | 334,863 |
| Edges | 925,872 |
| Type | Undirected, unweighted |
| Description | Products as nodes; co-purchased products connected by edges |

---
