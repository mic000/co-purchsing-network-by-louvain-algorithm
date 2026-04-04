import math
from typing import List, Tuple, Dict
from types_and_utils import (
  NodeID, ClusterID, Clustering, ClusterVolumes, EdgeList,
  is_active, count_edges_to_clusters, compute_cluster_volumes
)

# Helper for Shannon Entropy: p * log2(p)
def nplogp(p: float) -> float:
  return p * math.log2(p) if p > 1e-15 else 0.0

def compute_cluster_cuts(adj_rdd, clusters_bc) -> Dict[ClusterID, int]:
  def count_external(record: Tuple[NodeID, List[NodeID]]) -> Tuple[ClusterID, int]:
    node, neighbors = record
    c = clusters_bc.value[node]
    external = sum(1 for u in neighbors if clusters_bc.value[u] != c)
    return (c, external)

  return dict(adj_rdd.map(count_external).reduceByKey(lambda a, b: a + b).collect())


def compute_best_move_map_distributed(
  node: NodeID,
  neighbors: List[NodeID],
  clusters: Clustering,
  cluster_vol: ClusterVolumes,
  cluster_cut: Dict[ClusterID, int],
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

  edges_to_cluster = count_edges_to_clusters(neighbors, clusters)

  edges_to_current = edges_to_cluster.get(current_cluster, 0)

  # Current cluster stats if v leaves
  vol_c_minus_v = cluster_vol.get(current_cluster, 0) - deg_v

  edges_to_others = deg_v - edges_to_current
  cut_c_minus_v = cluster_cut.get(current_cluster, 0) - edges_to_others + edges_to_current

  vol_c = cluster_vol.get(current_cluster, 0)
  cut_c = cluster_cut.get(current_cluster, 0)

  # Per-cluster terms from the map equation (Section 2)
  before_current = -2 * nplogp(cut_c / m2) + nplogp((cut_c + vol_c) / m2)
  after_current = -2 * nplogp(cut_c_minus_v / m2) + nplogp((cut_c_minus_v + vol_c_minus_v) / m2)

  best_cluster = current_cluster
  best_delta = 0.0

  for target_cluster, edges_to_target in edges_to_cluster.items():
    if target_cluster == current_cluster:
      continue

    vol_t = cluster_vol.get(target_cluster, 0)
    cut_t = cluster_cut.get(target_cluster, 0)

    vol_t_plus_v = vol_t + deg_v
    cut_t_plus_v = cut_t + (deg_v - edges_to_target) - edges_to_target

    before_target = -2 * nplogp(cut_t / m2) + nplogp((cut_t + vol_t) / m2)
    after_target = -2 * nplogp(cut_t_plus_v / m2) + nplogp((cut_t_plus_v + vol_t_plus_v) / m2)

    delta_current = before_current - after_current
    delta_target = before_target - after_target

    old_sum_contribution = cut_c + cut_t
    new_sum_contribution = cut_c_minus_v + cut_t_plus_v
    delta_index = nplogp(old_sum_contribution / m2) - nplogp(new_sum_contribution / m2)

    delta = delta_current + delta_target + delta_index

    if delta > best_delta:
      best_delta = delta
      best_cluster = target_cluster

  return (node, best_cluster)

def dslm_local_moving_map_pyspark(
  sc,
  edges: EdgeList,
  max_rounds: int = 8,
  num_sub_rounds: int = 4,
  seed: int = 42
) -> Clustering:
  edge_rdd = sc.parallelize(edges)
  adj_rdd = edge_rdd.flatMap(lambda e: [(e[0], e[1]),  
  (e[1], e[0])]) \
    .groupByKey() \
    .mapValues(list) \
    .cache()

  node_degrees = dict(adj_rdd.mapValues(len).collect())
  total_volume = sum(node_degrees.values())

  clusters = {node: node for node in node_degrees}

  print(f"PySpark DSLM-Map: {len(node_degrees)} nodes, vol(V)={total_volume}")

  for round_num in range(max_rounds):
    moved_count = 0

    # Precompute cluster volumes and cuts for this round
    cluster_vol = compute_cluster_volumes(clusters, node_degrees)

    clusters_bc_temp = sc.broadcast(clusters)
    cluster_cut = compute_cluster_cuts(adj_rdd, clusters_bc_temp)
    clusters_bc_temp.unpersist()

    for sub_round in range(num_sub_rounds):
      clusters_bc = sc.broadcast(clusters)
      cluster_vol_bc = sc.broadcast(cluster_vol)
      cluster_cut_bc = sc.broadcast(cluster_cut)
      total_vol_bc = sc.broadcast(total_volume)

      rn = round_num
      sr = sub_round
      nsr = num_sub_rounds
      sd = seed

      def process_node(
        record: Tuple[NodeID, List[NodeID]]
      ) -> Tuple[NodeID, ClusterID]:
        """Wrapper that calls compute_best_move_map_distributed with broadcast data."""
        node, neighbors = record
        return compute_best_move_map_distributed(
          node=node,
          neighbors=neighbors,
          clusters=clusters_bc.value,
          cluster_vol=cluster_vol_bc.value,
          cluster_cut=cluster_cut_bc.value,
          total_volume=total_vol_bc.value,
          round_num=rn,
          sub_round=sr,
          num_sub_rounds=nsr,
          seed=sd
        )

      new_assignments = dict(adj_rdd.map(process_node).collect())

      for node, new_c in new_assignments.items():
        old_c = clusters[node]
        if new_c != old_c:
          moved_count += 1
          deg = node_degrees[node]
          cluster_vol[old_c] -= deg
          cluster_vol[new_c] += deg

      clusters = new_assignments

      clusters_bc.unpersist()
      cluster_vol_bc.unpersist()
      cluster_cut_bc.unpersist()
      total_vol_bc.unpersist()

    print(f"  Round {round_num + 1}: {moved_count} nodes moved")

    if moved_count == 0:
      print(f"  Converged after {round_num + 1} rounds")
      break

  return clusters