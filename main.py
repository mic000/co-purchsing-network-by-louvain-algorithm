import networkx as nx
import time
import os
import sys
import community as community_louvain
from collections import Counter
from infomap import Infomap
from plotting import algorithm_comparison_graph, community_distribution, detected_community
from sklearn.metrics import adjusted_rand_score
from dslm_mod_python import dslm_local_moving_mod_python
from dslm_map_python import dslm_local_moving_map_python
from dslm_mod_pyspark import dslm_local_moving_mod_pyspark
from dslm_map_pyspark import dslm_local_moving_map_pyspark
from pyspark import SparkContext, SparkConf

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == "__main__":
  edge_file = "data/com-amazon.ungraph.txt"
  G = nx.Graph()
  with open(edge_file, "r") as f:
    for line in f:
      if line.startswith("#"):
        continue
      u, v = map(int, line.strip().split())
      G.add_edge(u, v)

  # Load ground truth communities
  ground_truth = {}
  with open("data/com-amazon.all.dedup.cmty.txt") as f:
    for i, line in enumerate(f):
      for node in line.strip().split():
        node = int(node)
        if node not in ground_truth:
          ground_truth[node] = i

  # chosen the max number connected nodes
  largest_cc_nodes = max(nx.connected_components(G), key=len)
  G = G.subgraph(largest_cc_nodes).copy()

  print("\nUsing largest connected component:")
  print("Nodes:", G.number_of_nodes())
  print("Edges:", G.number_of_edges())

  # setup pyspark
  conf = SparkConf().setAppName("DSLM").setMaster("local[*]")
  sc = SparkContext(conf=conf)

  # --- Run plain Python version on small subgraph ---
  print("\n--- Testing on small subgraph ---")
  small_nodes= list(G.nodes())[:500]
  small_G = G.subgraph(small_nodes).copy()

  if not nx.is_connected(small_G):
    test_cc = max(nx.connected_components(small_G), key=len)
    test_G = small_G.subgraph(test_cc).copy()
  else:
    test_G = small_G

  print(f"Test graph: {test_G.number_of_nodes()} nodes, {test_G.number_of_edges()} edges")

  print("\nUsing Pyspark Mod")
  start = time.time()
  result = dslm_local_moving_mod_pyspark(sc, list(test_G.edges()))
  elapsed = time.time() - start
  num_communities = len(set(result.values()))
  Q = community_louvain.modularity(result, test_G)
  print(f"\nResult: {num_communities} communities, Q = {Q:.4f}, Time = {elapsed:.4f}s")

  print("\nUsing Python Mod")
  start_2 = time.time()
  result_2 = dslm_local_moving_mod_python(test_G)
  elapsed_2 = time.time() - start_2
  num_communities_2 = len(set(result_2.values()))
  Q_2 = community_louvain.modularity(result_2, test_G)
  print(f"\nResult: {num_communities_2} communities, Q = {Q_2:.4f}, Time = {elapsed_2:.4f}s")

  print("\nUsing Pyspark Map")
  start_3 = time.time()
  result_3 = dslm_local_moving_map_pyspark(sc, list(test_G.edges()))
  elapsed_3 = time.time() - start_3
  num_communities_3 = len(set(result_3.values()))
  Q_3 = community_louvain.modularity(result_3, test_G)
  print(f"\nResult: {num_communities_3} communities, Q = {Q_3:.4f}, Time = {elapsed_3:.4f}s")

  print("\nUsing Python Map")
  start_4 = time.time()
  result_4 = dslm_local_moving_map_python(test_G)
  elapsed_4 = time.time() - start_4
  num_communities_4 = len(set(result_4.values()))
  Q_4 = community_louvain.modularity(result_4, test_G)
  print(f"\nResult: {num_communities_4} communities, Q = {Q_4:.4f}, Time = {elapsed_4:.4f}s")
  # --- End of testing on small subgraph ---


  # ----- Calculate with All nodes -----
  # --- Sequential Louvain (Baseline for DSLM Mod) ---
  print("\n=== Sequential Louvain (Library Baseline) ===")
  start_louvain = time.time()
  partition_louvain = community_louvain.best_partition(G, random_state=42)
  time_louvain = time.time() - start_louvain

  Q_louvain = community_louvain.modularity(partition_louvain, G)
  num_comm_louvain = len(set(partition_louvain.values()))
  sizes_louvain = Counter(partition_louvain.values())

  print(f"Runtime: {time_louvain:.4f}s")
  print(f"Communities: {num_comm_louvain}")
  print(f"Modularity Q: {Q_louvain:.4f}")
  # ----- ground truth calculation -----
  # Only compare nodes that exist in both
  common_nodes = sorted(set(ground_truth.keys()) & set(partition_louvain.keys()))
  true_labels = [ground_truth[n] for n in common_nodes]
  pred_labels = [partition_louvain[n] for n in common_nodes]

  ari_louvain = adjusted_rand_score(true_labels, pred_labels)
  print(f"ARI with ground truth of DSLM Map : {ari_louvain:.4f}")
  print(f"Largest community: {max(sizes_louvain.values())}")
  print(f"Smallest community: {min(sizes_louvain.values())}")
  print(f"Average community size: {sum(sizes_louvain.values()) / len(sizes_louvain):.1f}")
  # --- End of Sequential Louvain ---

  # --- Sequential Infomap (Baseline for DSLM Map) ---
  print("\n=== Sequential Infomap (Library Baseline) ===")
  start_infomap = time.time()
  im = Infomap("--two-level")
  for u, v in G.edges():
    im.add_link(u, v)
  im.run()
  time_infomap = time.time() - start_infomap

  partition_infomap = {}
  for node in im.nodes:
    partition_infomap[node.node_id] = node.module_id

  num_comm_infomap = len(set(partition_infomap.values()))
  sizes_infomap = Counter(partition_infomap.values())
  Q_infomap = community_louvain.modularity(partition_infomap, G)

  print(f"Runtime: {time_infomap:.4f}s")
  print(f"Communities: {num_comm_infomap}")
  print(f"Modularity Q: {Q_infomap:.4f}")
  # ----- ground truth calculation -----
  # Only compare nodes that exist in both
  common_nodes = sorted(set(ground_truth.keys()) & set(partition_infomap.keys()))
  true_labels = [ground_truth[n] for n in common_nodes]
  pred_labels = [partition_infomap[n] for n in common_nodes]

  ari_infomap = adjusted_rand_score(true_labels, pred_labels)
  print(f"ARI with ground truth of DSLM Map : {ari_infomap:.4f}")
  print(f"Largest community: {max(sizes_infomap.values())}")
  print(f"Smallest community: {min(sizes_infomap.values())}")
  print(f"Average community size: {sum(sizes_infomap.values()) / len(sizes_infomap):.1f}")

  # --- Pyspark Mod --
  print("\n--- Starting DSLM Mod ---")
  edges = list(G.edges())
  start_dslm_mod = time.time()
  result_dslm_mod = dslm_local_moving_mod_pyspark(sc, edges)
  time_dslm_mod = time.time() - start
  num_comm_dslm_mod = len(set(result_dslm_mod.values()))
  Q_dslm_mod = community_louvain.modularity(result_dslm_mod, G)
  print(f"PySpark: {num_comm_dslm_mod} communities, Q = {Q_dslm_mod:.4f}, Time = {time_dslm_mod:.2f}s")
  
  # -- ground truth calculation --
  # Only compare nodes that exist in both
  common_nodes = sorted(set(ground_truth.keys()) & set(result_dslm_mod.keys()))
  true_labels = [ground_truth[n] for n in common_nodes]
  pred_labels = [result_dslm_mod[n] for n in common_nodes]

  ari_mod = adjusted_rand_score(true_labels, pred_labels)
  print(f"ARI with ground truth of DSLM Mod : {ari_mod:.4f}")

  # -- clustering granuality --
  sizes_dslm_mod = Counter(result_dslm_mod.values())
  print(f"Smallest community: {min(sizes_dslm_mod.values())}")
  print(f"Largest community: {max(sizes_dslm_mod.values())}")
  print(f"Average community size: {sum(sizes_dslm_mod.values()) / len(sizes_dslm_mod):.1f}")
  print(f"Median community size: {sorted(sizes_dslm_mod.values())[len(sizes_dslm_mod)//2]}")
  # --- end of Pyspark Mod ---

  # --- Pyspark Map ---
  print("\n--- Starting DSLM Map ---")
  edges = list(G.edges())
  start_dslm_map = time.time()
  result_dslm_map = dslm_local_moving_map_pyspark(sc, edges)
  time_dslm_map = time.time() - start
  num_comm_dslm_map = len(set(result_dslm_map.values()))
  Q_dslm_map = community_louvain.modularity(result_dslm_map, G)
  print(f"PySpark: {num_comm_dslm_map} communities, Q = {Q_dslm_map:.4f}, Time = {time_dslm_map:.2f}s")

  # ----- ground truth calculation -----
  # Only compare nodes that exist in both
  common_nodes = sorted(set(ground_truth.keys()) & set(result_dslm_map.keys()))
  true_labels = [ground_truth[n] for n in common_nodes]
  pred_labels = [result_dslm_map[n] for n in common_nodes]

  ari_map = adjusted_rand_score(true_labels, pred_labels)
  print(f"ARI with ground truth of DSLM Map : {ari_map:.4f}")

  # clustering granuality
  sizes_dslm_map = Counter(result_dslm_map.values())
  print(f"Smallest community: {min(sizes_dslm_map.values())}")
  print(f"Largest community: {max(sizes_dslm_map.values())}")
  print(f"Average community size: {sum(sizes_dslm_map.values()) / len(sizes_dslm_map):.1f}")
  print(f"Median community size: {sorted(sizes_dslm_map.values())[len(sizes_dslm_map)//2]}")
  # --- end of Pyspark Map ---

  # Plotting
  # comparison graph
  algorithm_comparison_graph(time_louvain,num_comm_louvain,Q_louvain,sizes_louvain, time_infomap, num_comm_infomap, Q_infomap, sizes_infomap, time_dslm_mod, num_comm_dslm_mod, Q_dslm_mod, sizes_dslm_mod, time_dslm_map, num_comm_dslm_map, Q_dslm_map, sizes_dslm_map)

  # COMMUNITY SIZE DISTRIBUTION — ALL 4 METHODS
  community_distribution(sizes_louvain, sizes_infomap, sizes_dslm_mod, sizes_dslm_map)

  # Community Detection
  detected_community(result_dslm_mod, result_dslm_map, partition_louvain, partition_infomap, G)

  # stop spark context
  sc.stop()
