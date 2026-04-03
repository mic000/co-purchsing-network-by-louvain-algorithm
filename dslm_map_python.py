import networkx as nx
import math
import random
from typing import Dict, List, Set
from types_and_utils import NodeID, ClusterID, Clustering, is_active

def nplogp(p: float) -> float:
  return p * math.log2(p) if p > 1e-15 else 0.0

def compute_map_equation_delta(
  node: NodeID, 
  target_cluster: ClusterID, 
  clusters: Clustering, 
  G: nx.Graph
) -> float:
  m2 = float(2 * G.number_of_edges())
  current_cluster = clusters[node]
    
  # Node metadata
  neighbors = list(G.neighbors(node))
  deg_v = G.degree(node)
    
  # Calculate edges to clusters
  edges_to_target = sum(1 for u in neighbors if clusters[u] == target_cluster)
    
  # Target Cluster Stats
  nodes_in_target = [u for u in G.nodes() if clusters[u] == target_cluster]
  vol_t = sum(G.degree(u) for u in nodes_in_target)
  # Cut_t is the sum of degrees of nodes in T minus edges internal to T
  internal_edges_t = sum(1 for u in nodes_in_target for v in G.neighbors(u) if clusters[v] == target_cluster)
  cut_t = vol_t - internal_edges_t

  # Calculate change if v joins target
  # New cut = old_cut + (edges from v to outside T) - (edges from v to inside T)
  cut_t_plus_v = cut_t + (deg_v - edges_to_target) - edges_to_target
  vol_t_plus_v = vol_t + deg_v

  # Map Equation Delta: Change in entropy contribution
  # We want to MINIMIZE code length, so a positive delta here means a reduction.
  term_before = -2 * nplogp(cut_t / m2) + nplogp((cut_t + vol_t) / m2)
  term_after = -2 * nplogp(cut_t_plus_v / m2) + nplogp((cut_t_plus_v + vol_t_plus_v) / m2)
    
  return term_before - term_after

def dslm_local_moving_map_python(
  G: nx.Graph,
  max_rounds: int = 8,
  num_sub_rounds: int = 4,
  seed: int = 42
) -> Clustering:
  random.seed(seed)
  clusters: Clustering = {node: node for node in G.nodes()}

  for round_num in range(max_rounds):
    moved_count = 0
    for sub_round in range(num_sub_rounds):
    # Synchronous update: decisions are made based on the state at the start of sub-round
      active_nodes = [v for v in G.nodes() if is_active(v, round_num, sub_round, num_sub_rounds, seed)]
      
      new_moves = {}

      for v in active_nodes:
        current_C = clusters[v]
        neighbor_clusters = set(clusters[u] for u in G.neighbors(v))
                
        best_C = current_C
        max_delta = 0.0

        for target_C in neighbor_clusters:
          if target_C == current_C: continue
          delta = compute_map_equation_delta(v, target_C, clusters, G)
          if delta > max_delta:
            max_delta = delta
            best_C = target_C
                
        if best_C != current_C:
          new_moves[v] = best_C

      # Apply moves
      for v, new_C in new_moves.items():
        clusters[v] = new_C
        moved_count += 1

    print(f"  Round {round_num + 1}: {moved_count} nodes moved")

    if moved_count == 0:
      print(f"  Converged after {round_num + 1} rounds")
      break

  return clusters