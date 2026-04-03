from types_and_utils import EdgeList, NodeID, NodeDegrees, Clustering, ClusterVolumes, ClusterID, NeighborClusterEdges, is_active, count_edges_to_clusters, compute_cluster_volumes
from typing import List, Tuple

# Same logic as the Python version's inner loop but for one node, using broadcast data (cluster volumes, total volume) instead of scanning the whole graph. This function runs in parallel across all workers.
def compute_best_move_distributed(
  node: NodeID,
  neighbors: List[NodeID],
  clusters: Clustering,
  cluster_vol: ClusterVolumes,
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

  edges_to_current = edges_to_cluster.get(current_cluster, 0)
  vol_current_without_v = cluster_vol.get(current_cluster, 0) - deg_v

  best_cluster = current_cluster
  best_delta = 0.0

  for target_cluster, cut_v_target in edges_to_cluster.items():
    if target_cluster == current_cluster:
      continue

    vol_target = cluster_vol.get(target_cluster, 0)

    # this is the delta_q for modularity
    delta = (cut_v_target - edges_to_current) / m2 \
      - deg_v * (vol_target - vol_current_without_v) / (m2 * m2)
    
    delta *= 2

    if delta > best_delta:
      best_delta = delta
      best_cluster = target_cluster

  return (node, best_cluster)

# Orchestrates the distributed version: builds the adjacency RDD, broadcasts clustering state each sub-round, runs compute_best_move_distributed on all nodes via adj_rdd.map(), collects results, and updates.
def dslm_local_moving_mod_pyspark(
    sc,  # pyspark.SparkContext
    edges: EdgeList,
    max_rounds: int = 8,
    num_sub_rounds: int = 4,
    seed: int = 42
) -> Clustering:
  edge_rdd = sc.parallelize(edges)
  adj_rdd = edge_rdd.flatMap(lambda e: [(e[0], e[1]), (e[1], e[0])]) \
    .groupByKey() \
    .mapValues(list) \
    .cache()

  node_degrees = dict(adj_rdd.mapValues(len).collect())
  total_volume = sum(node_degrees.values())

  clusters = {node: node for node in node_degrees}

  print(f"PySpark DSLM: {len(node_degrees)} nodes, vol(V)={total_volume}")

  for round_num in range(max_rounds):
    moved_count: int = 0

    cluster_vol = compute_cluster_volumes(clusters, node_degrees)

    for sub_round in range(num_sub_rounds):
      clusters_bc = sc.broadcast(clusters)
      cluster_vol_bc = sc.broadcast(cluster_vol)
      total_vol_bc = sc.broadcast(total_volume)

      rn = round_num # round number
      sr = sub_round # sub round # at the moment
      nsr = num_sub_rounds # total number of sub round
      sd = seed # seed

      def process_node(record: Tuple[NodeID, List[NodeID]]) -> Tuple[NodeID, ClusterID]:
        node, neighbors = record
        return compute_best_move_distributed(
          node=node,
          neighbors=neighbors,
          clusters=clusters_bc.value,
          cluster_vol=cluster_vol_bc.value,
          total_volume=total_vol_bc.value,
          round_num=rn,
          sub_round=sr,
          num_sub_rounds=nsr,
          seed=sd
        )

      new_assignments: Clustering = dict(adj_rdd.map(process_node).collect())

      for node, new_c in new_assignments.items():
        old_c: ClusterID = clusters[node]
        if new_c != old_c:
          moved_count += 1
          deg: int = node_degrees[node]
          cluster_vol[old_c] -= deg
          cluster_vol[new_c] += deg

      clusters = new_assignments

      clusters_bc.unpersist()
      cluster_vol_bc.unpersist()
      total_vol_bc.unpersist()

    print(f"  Round {round_num + 1}: {moved_count} nodes moved")

    if moved_count == 0:
      print(f"  Converged after {round_num + 1} rounds")
      break

  return clusters