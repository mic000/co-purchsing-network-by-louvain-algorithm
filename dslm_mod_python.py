from types_and_utils import NodeID, ClusterID, Clustering, is_active
from typing import List, Dict, Set
import networkx as nx
import random

# Calculates how much Q would change if one node moved to a different cluster. This is the delta that decides whether a move is worth it.
def compute_modularity_gain(
  node: NodeID,
  target_cluster: ClusterID,
  clusters: Clustering,
  G: nx.Graph
) -> float:
  vol_V: int = 2 * G.number_of_edges()
  if vol_V == 0:
    return 0.0

  current_cluster: ClusterID = clusters[node]
  if current_cluster == target_cluster:
    return 0.0 

  deg_v: int = G.degree(node)

  cut_v_target: int = sum(1 for u in G.neighbors(node) if clusters[u] == target_cluster)
  cut_v_current: int = sum(1 for u in G.neighbors(node) if clusters[u] == current_cluster)

  vol_current: int = sum(G.degree(u) for u in G.nodes() if clusters[u] == current_cluster)
  vol_current_without_v: int = vol_current - deg_v

  vol_target: int = sum(G.degree(u) for u in G.nodes() if clusters[u] == target_cluster)

  delta_Q: float = (cut_v_target - cut_v_current) / vol_V \
    - deg_v * (vol_target - vol_current_without_v) / (vol_V * vol_V)

  return 2 * delta_Q

# The full local moving phase: runs up to 8 rounds, each with 4 sub-rounds, where active nodes pick their best cluster and move simultaneously.
def dslm_local_moving_mod_python(
  G: nx.Graph,
  max_rounds: int = 8,
  num_sub_rounds: int = 4,
  seed: int = 42
) -> Clustering:
  random.seed(seed)

  clusters: Clustering = {node: node for node in G.nodes()}

  for round_num in range(max_rounds):
    moved_count: int = 0

    for sub_round in range(num_sub_rounds):
      active_nodes: List[NodeID] = [v for v in G.nodes() if is_active(v, round_num, sub_round, num_sub_rounds, seed)]

      moves: Dict[NodeID, ClusterID] = {}

      for v in active_nodes:
        current_cluster: ClusterID = clusters[v]
        best_cluster: ClusterID = current_cluster
        best_delta: float = 0.0

        neighbor_clusters: Set[ClusterID] = set(clusters[u] for u in G.neighbors(v) if clusters[u] != current_cluster)

        for target_C in neighbor_clusters:
          delta: float = compute_modularity_gain(v, target_C, clusters, G)
          if delta > best_delta:
            best_delta = delta
            best_cluster = target_C

        if best_cluster != current_cluster:
          moves[v] = best_cluster

      for v, new_cluster in moves.items():
        clusters[v] = new_cluster
        moved_count += 1

    print(f"  Round {round_num + 1}: {moved_count} nodes moved")

    if moved_count == 0:
      print(f"  Converged after {round_num + 1} rounds")
      break

  return clusters