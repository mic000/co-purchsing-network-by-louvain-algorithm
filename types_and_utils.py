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

# modularity of clustering (C)
def compute_modularity(
  clusters: Clustering,
  G: nx.Graph
) -> float:
  # Section 2
  # original formula: Q(C) = sum_C [ (vol(C) - cut(C)) / vol(V) - (vol(C) / vol(V))^2 ]

  vol_V: int = 2 * G.number_of_edges()
  if vol_V == 0:
    return 0.0

  community_ids: Set[ClusterID] = set(clusters.values())
  Q: float = 0.0

  for c in community_ids:
    nodes_in_c: List[NodeID] = [v for v in G.nodes() if clusters[v] == c]
    vol_c: int = sum(G.degree(v) for v in nodes_in_c)

    # vol(C) - cut(C)
    internal_edges: int = sum(1 for v in nodes_in_c for u in G.neighbors(v) if clusters[u] == c)

    Q += internal_edges / vol_V - (vol_c / vol_V) ** 2

  return Q
  
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

# Flips a node→cluster dict into a cluster→node list dict. Just a helper for visualization and analysis.
def invert_partition(partition: Clustering) -> Dict[ClusterID, List[NodeID]]:
  comm_to_nodes: DefaultDict[ClusterID, List[NodeID]] = defaultdict(list)
  for node, comm_id in partition.items():
    comm_to_nodes[comm_id].append(node)
  return dict(comm_to_nodes)