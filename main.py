import networkx as nx
import time
import os
import sys
import community as community_louvain
# from types_and_utils import compute_modularity
from dslm_mod_python import dslm_local_moving_mod_python
from dslm_map_python import dslm_local_moving_map_python
from dslm_mod_pyspark import dslm_local_moving_mod_pyspark
from dslm_map_pyspark import dslm_local_moving_map_pyspark
from pyspark import SparkContext, SparkConf

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == "__main__":
  edge_file = "com-amazon.ungraph.txt"
  G = nx.Graph()
  with open(edge_file, "r") as f:
    for line in f:
      if line.startswith("#"):
        continue
      u, v = map(int, line.strip().split())
      G.add_edge(u, v)

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

  # Pyspark Mod -> calculate all
  print("\n--- Starting DSLM Mod ---")
  edges = list(G.edges())
  start = time.time()
  result_spark = dslm_local_moving_mod_pyspark(sc, edges)
  elapsed = time.time() - start
  num_communities = len(set(result_spark.values()))
  # Q_spark = compute_modularity(result_spark, G)
  Q_spark = community_louvain.modularity(result_spark, G)
  print(f"PySpark: {num_communities} communities, Q = {Q_spark:.4f}, Time = {elapsed:.2f}s")
  # end of :Pyspark Mod

  # Pyspark Map -> calculate all
  print("\n--- Starting DSLM Map ---")
  edges = list(G.edges())
  start = time.time()
  result_spark = dslm_local_moving_map_pyspark(sc, edges)
  elapsed = time.time() - start
  num_communities = len(set(result_spark.values()))
  # Q_spark = compute_modularity(result_spark, G)
  Q_spark = community_louvain.modularity(result_spark, G)
  print(f"PySpark: {num_communities} communities, Q = {Q_spark:.4f}, Time = {elapsed:.2f}s")
  # end of :Pyspark Map

  # stop spark context
  sc.stop()