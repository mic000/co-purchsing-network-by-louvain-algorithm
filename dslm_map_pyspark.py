import math
from typing import List, Tuple, Dict
from types_and_utils import (
  NodeID, ClusterID, Clustering, ClusterVolumes, 
  NeighborClusterEdges, is_active, count_edges_to_clusters
)

# Helper for Shannon Entropy: p * log2(p)
def nplogp(p: float) -> float:
  return p * math.log2(p) if p > 0 else 0.0

def compute_best_move_map_distributed(
  node: NodeID,
  neighbors: List[NodeID],
  clusters: Clustering,
  cluster_vol: ClusterVolumes,
  cluster_cut: Dict[ClusterID, int], # New requirement for Map
  total_volume: int,
  round_num: int,
  sub_round: int,
  num_sub_rounds: int,
  seed: int
) -> Tuple[NodeID, ClusterID]:  
  if not is_active(node, round_num, sub_round, num_sub_rounds, seed):
    return (node, clusters[node])

  current_cluster = clusters[node]
  deg_v = len(neighbors)
  m2 = total_volume
    
  edges_to_cluster: NeighborClusterEdges = count_edges_to_clusters(neighbors, clusters)
    
  # 1. Calculate current state properties for node v
  edges_to_current = edges_to_cluster.get(current_cluster, 0)
  edges_to_others = deg_v - edges_to_current
    
  # Volume and Cut of current cluster if v leaves
  vol_c_minus_v = cluster_vol.get(current_cluster, 0) - deg_v
  # If v leaves, the cluster cut changes: 
  # it loses edges that were going from v to outside the cluster, 
  # but gains edges that were going from v to inside the cluster.
  cut_c_minus_v = cluster_cut.get(current_cluster, 0) - edges_to_others + edges_to_current

  best_cluster = current_cluster
  min_codelength_delta = 0.0 # We want to MINIMIZE codelength

  # 2. Iterate through neighbor clusters to find the best move
  for target_cluster, edges_to_target in edges_to_cluster.items():
    if target_cluster == current_cluster:
      continue
        
    vol_t = cluster_vol.get(target_cluster, 0)
    cut_t = cluster_cut.get(target_cluster, 0)

    # Map Equation Delta (Simplified for local moving)
    # We calculate the change in entropy components:
    # q_out * log(q_out) and (q_out + vol) * log(q_out + vol)
        
    # This is a simplified transition logic:
    # Gain = [Entropy of Target with v] - [Entropy of Target without v]
    # In Map Equation, we look for the move that reduces the overall 
    # description length L(P).
        
    # New cut for target if v joins
    cut_t_plus_v = cut_t + (deg_v - edges_to_target) - edges_to_target
    vol_t_plus_v = vol_t + deg_v

    # Simplified Map Delta (Change in index codebook + module codebook)
    # Higher delta here = better reduction in code length
    delta = (nplogp(cut_t / m2) + nplogp((cut_t + vol_t) / m2)) - \
      (nplogp(cut_t_plus_v / m2) + nplogp((cut_t_plus_v + vol_t_plus_v) / m2))

    if delta > min_codelength_delta:
      min_codelength_delta = delta
      best_cluster = target_cluster

  return (node, best_cluster)

def dslm_local_moving_map_pyspark(
  sc, 
  edges,
  max_rounds=8, 
  num_sub_rounds=4, 
  seed=42
):
  edges_rdd = sc.parallelize(edges)
  adj_rdd = edges_rdd.flatMap(lambda x: [(x[0], x[1]), (x[1], x[0])]) \
    .groupByKey() \
    .mapValues(list) \
    .cache()
  node_degrees = adj_rdd.mapValues(len).collectAsMap()
  total_volume = sum(node_degrees.values())
  clusters = {node: node for node in node_degrees.keys()}
  # Map Equation needs Cluster Volume AND Cluster Cut (sum of edges leaving the cluster)
  cluster_vol = {n: d for n, d in node_degrees.items()}
  cluster_cut = {n: d for n, d in node_degrees.items()} 

  for round_num in range(max_rounds):
    moved_count = 0

    for sub_round in range(num_sub_rounds):
      clusters_bc = sc.broadcast(clusters)
      cluster_vol_bc = sc.broadcast(cluster_vol)
      cluster_cut_bc = sc.broadcast(cluster_cut)

      def process_node(record):
        node, neighbors = record
        return compute_best_move_map_distributed(
          node, neighbors, clusters_bc.value, 
          cluster_vol_bc.value, cluster_cut_bc.value,
          total_volume, round_num, sub_round, num_sub_rounds, seed
        )

      new_assignments = dict(adj_rdd.map(process_node).collect())

      for node, new_c in new_assignments.items():
        old_c: ClusterID = clusters[node]
        
        if new_c != old_c:
          moved_count += 1 
          deg_v = node_degrees[node]

          cluster_vol[old_c] -= deg_v
          cluster_vol[new_c] += deg_v
              
          # Update Volumes
          deg: int = node_degrees[node]
          cluster_vol[old_c] -= deg
          cluster_vol[new_c] += deg
              
          # Update Assignment
          clusters[node] = new_c

      clusters_bc.unpersist()
      cluster_vol_bc.unpersist()
      cluster_cut_bc.unpersist()

    print(f"  Round {round_num + 1}: {moved_count} nodes moved")

    if moved_count == 0:
      print(f"  Converged after {round_num + 1} rounds")
      break
  
  return clusters