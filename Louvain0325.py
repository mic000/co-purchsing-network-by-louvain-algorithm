import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter, defaultdict
import community as community_louvain
from infomap import Infomap

# ===== 1. Load graph =====
edge_file = "com-amazon.ungraph.txt"   # file name

G = nx.Graph()
with open(edge_file, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        u, v = map(int, line.strip().split())
        G.add_edge(u, v)

print("Graph loaded.")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected components:", nx.number_connected_components(G))

# chosen the max number connected nodes
largest_cc_nodes = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc_nodes).copy()

print("\nUsing largest connected component:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# ===== 2. Modularity-based clustering (proxy for DSLM-Mod) =====
start = time.time()
partition_mod = community_louvain.best_partition(G)
time_mod = time.time() - start

modularity_score = community_louvain.modularity(partition_mod, G)
num_comm_mod = len(set(partition_mod.values()))
sizes_mod = Counter(partition_mod.values())

print("\n=== Modularity-based result ===")
print("Runtime (s):", round(time_mod, 4))
print("Number of communities:", num_comm_mod)
print("Modularity:", round(modularity_score, 4))
print("Largest community size:", max(sizes_mod.values()))


# ===== 3. Map-equation-based clustering (proxy for DSLM-Map) =====
start = time.time()
im = Infomap("--two-level")
for u, v in G.edges():
    im.add_link(u, v)

im.run()
time_map = time.time() - start

partition_map = {}
for node in im.nodes:
    partition_map[node.node_id] = node.module_id

num_comm_map = len(set(partition_map.values()))
sizes_map = Counter(partition_map.values())

print("\n=== Map-equation-based result ===")
print("Runtime (s):", round(time_map, 4))
print("Number of communities:", num_comm_map)
print("Largest community size:", max(sizes_map.values()))


# ===== 4. Compare summary =====
print("\n=== Comparison Summary ===")
print(f"{'Method':<20}{'Runtime(s)':<15}{'#Communities':<15}{'Largest Community':<20}")
print("-" * 70)
print(f"{'Modularity-based':<20}{time_mod:<15.4f}{num_comm_mod:<15}{max(sizes_mod.values()):<20}")
print(f"{'Map-equation-based':<20}{time_map:<15.4f}{num_comm_map:<15}{max(sizes_map.values()):<20}")


# ===== 5. Community size distributions =====
plt.figure(figsize=(8, 5))
plt.hist(list(sizes_mod.values()), bins=30, alpha=0.7)
plt.title("Community Size Distribution - Modularity-based")
plt.xlabel("Community Size")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(list(sizes_map.values()), bins=30, alpha=0.7)
plt.title("Community Size Distribution - Map-equation-based")
plt.xlabel("Community Size")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ===== 6. Helper: convert partition to community -> node list =====
def invert_partition(partition):
    comm_to_nodes = defaultdict(list)
    for node, comm_id in partition.items():
        comm_to_nodes[comm_id].append(node)
    return comm_to_nodes

comm_nodes_mod = invert_partition(partition_mod)
comm_nodes_map = invert_partition(partition_map)

# chosen a medium-sized community to plot out
def pick_medium_community(comm_to_nodes, min_size=20, max_size=80):
    for comm_id, nodes in comm_to_nodes.items():
        if min_size <= len(nodes) <= max_size:
            return comm_id, nodes
    # If not found, return to the largest community
    comm_id = max(comm_to_nodes, key=lambda c: len(comm_to_nodes[c]))
    return comm_id, comm_to_nodes[comm_id]

# Modularity-based
comm_id_mod, nodes_mod = pick_medium_community(comm_nodes_mod)
subG_mod = G.subgraph(nodes_mod).copy()

print("Modularity-based chosen community:", comm_id_mod)
print("Nodes:", subG_mod.number_of_nodes(), "Edges:", subG_mod.number_of_edges())

fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(subG_mod, seed=42)
nx.draw_networkx_nodes(subG_mod, pos, ax=ax, node_size=40)
nx.draw_networkx_edges(subG_mod, pos, ax=ax, alpha=0.4)
ax.set_title(f"Detected Community (Modularity-based), ID={comm_id_mod}")
ax.set_axis_off()
plt.tight_layout()
plt.show()

# Also draw the map-equation version:
comm_id_map, nodes_map = pick_medium_community(comm_nodes_map)
subG_map = G.subgraph(nodes_map).copy()

print("Map-equation-based chosen community:", comm_id_map)
print("Nodes:", subG_map.number_of_nodes(), "Edges:", subG_map.number_of_edges())

fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(subG_map, seed=42)
nx.draw_networkx_nodes(subG_map, pos, ax=ax, node_size=40)
nx.draw_networkx_edges(subG_map, pos, ax=ax, alpha=0.4)
ax.set_title(f"Detected Community (Map-equation-based), ID={comm_id_map}")
ax.set_axis_off()
plt.tight_layout()
plt.show()