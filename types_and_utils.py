from collections import defaultdict
from typing import Dict, List, Set, Tuple, DefaultDict
import networkx as nx

NodeID = int
ClusterID = int
EdgeList = List[Tuple[NodeID, NodeID]]
Clustering = Dict[NodeID, ClusterID]
ClusterVolumes = Dict[ClusterID, int]
NeighborClusterEdges = Dict[ClusterID, int]
NodeDegrees = Dict[NodeID, int]

def is_active(
  node: NodeID,
  round_num: int,
  sub_round: int,
  num_sub_rounds: int,
  seed: int
) -> bool:
  return hash((node, round_num, seed)) % num_sub_rounds == sub_round

# Sums up the degrees of all nodes in each cluster. Used to precompute vol(C) so we don't recalculate it for every single node move.
def compute_cluster_volumes(
  clusters: Clustering,
  node_degrees: NodeDegrees
) -> ClusterVolumes:
  cluster_vol: DefaultDict[ClusterID, int] = defaultdict(int)
  for node, c in clusters.items():
    cluster_vol[c] += node_degrees[node]
  return dict(cluster_vol)

# Counts how many edges a node has going to each neighboring cluster. This is the "bid" data — tells the node how connected it is to each candidate cluster.
def count_edges_to_clusters(
  neighbors: List[NodeID],
  clusters: Clustering
) -> NeighborClusterEdges:
  edges_to_cluster: DefaultDict[ClusterID, int] = defaultdict(int)
  for u in neighbors:
    edges_to_cluster[clusters[u]] += 1
  return dict(edges_to_cluster)