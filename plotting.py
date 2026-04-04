import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def algorithm_comparison_graph(
  time_louvain,
  num_comm_louvain,
  Q_louvain,
  sizes_louvain,
  time_infomap,
  num_comm_infomap,
  Q_infomap,
  sizes_infomap,
  time_dslm_mod,
  num_comm_dslm_mod,
  Q_dslm_mod,
  sizes_dslm_mod,
  time_dslm_map,
  num_comm_dslm_map,
  Q_dslm_map,
  sizes_dslm_map,
):
  print("\n" + "=" * 85)
  print(f"{'Method':<20}{'Runtime(s)':<12}{'#Comm':<10}{'Q':<10}{'Largest':<10}{'Smallest':<10}{'Avg Size':<10}")
  print("-" * 85)
  print(f"{'Louvain (lib)':<20}{time_louvain:<12.2f}{num_comm_louvain:<10}{Q_louvain:<10.4f}{max(sizes_louvain.values()):<10}{min(sizes_louvain.values()):<10}{sum(sizes_louvain.values())/len(sizes_louvain):<10.1f}")
  print(f"{'Infomap (lib)':<20}{time_infomap:<12.2f}{num_comm_infomap:<10}{Q_infomap:<10.4f}{max(sizes_infomap.values()):<10}{min(sizes_infomap.values()):<10}{sum(sizes_infomap.values())/len(sizes_infomap):<10.1f}")
  print(f"{'DSLM-Mod (ours)':<20}{time_dslm_mod:<12.2f}{num_comm_dslm_mod:<10}{Q_dslm_mod:<10.4f}{max(sizes_dslm_mod.values()):<10}{min(sizes_dslm_mod.values()):<10}{sum(sizes_dslm_mod.values())/len(sizes_dslm_mod):<10.1f}")
  print(f"{'DSLM-Map (ours)':<20}{time_dslm_map:<12.2f}{num_comm_dslm_map:<10}{Q_dslm_map:<10.4f}{max(sizes_dslm_map.values()):<10}{min(sizes_dslm_map.values()):<10}{sum(sizes_dslm_map.values())/len(sizes_dslm_map):<10.1f}")
  print("=" * 85)

def community_distribution(sizes_louvain, sizes_infomap, sizes_dslm_mod, sizes_dslm_map):
  fig, axes = plt.subplots(2, 2, figsize=(12, 10))

  all_data = [
    (sizes_louvain, 'Louvain (Library)', '#85B7EB'),
    (sizes_infomap, 'Infomap (Library)', '#85B7EB'),
    (sizes_dslm_mod, 'DSLM-Mod (Ours)', '#5DCAA5'),
    (sizes_dslm_map, 'DSLM-Map (Ours)', '#5DCAA5'),
  ]

  for ax, (sizes, title, color) in zip(axes.flat, all_data):
    vals = list(sizes.values())
    ax.hist(vals, bins=50, alpha=0.8, color=color, edgecolor='black', linewidth=0.3)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Community Size')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    # Add stats as text
    ax.text(0.95, 0.95, f'n={len(sizes)}\nmax={max(vals)}\navg={sum(vals)/len(vals):.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

  plt.suptitle('Community Size Distribution Comparison', fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.show()

def detected_community(result_dslm_mod, result_dslm_map, result_louvain, result_infomap, G):
  def invert_partition(partition):
    comm_to_nodes = defaultdict(list)
    for node, comm_id in partition.items():
        comm_to_nodes[comm_id].append(node)
    return comm_to_nodes

  def pick_medium_community(comm_to_nodes, min_size=20, max_size=80):
    for comm_id, nodes in comm_to_nodes.items():
      if min_size <= len(nodes) <= max_size:
        return comm_id, nodes
    # fallback: pick largest under max_size
    candidates = {c: n for c, n in comm_to_nodes.items() if len(n) <= max_size}
    if candidates:
        comm_id = max(candidates, key=lambda c: len(candidates[c]))
        return comm_id, candidates[comm_id]
    # last resort: pick smallest overall
    comm_id = min(comm_to_nodes, key=lambda c: len(comm_to_nodes[c]))
    return comm_id, comm_to_nodes[comm_id]

  # --- DSLM-Mod community ---
  comm_nodes_dslm_mod = invert_partition(result_dslm_mod)
  cid_mod, nodes_mod = pick_medium_community(comm_nodes_dslm_mod)
  subG_mod = G.subgraph(nodes_mod).copy()

  # --- DSLM-Map community ---
  comm_nodes_dslm_map = invert_partition(result_dslm_map)
  cid_map, nodes_map = pick_medium_community(comm_nodes_dslm_map)
  subG_map = G.subgraph(nodes_map).copy()

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

  pos1 = nx.spring_layout(subG_mod, seed=42)
  nx.draw_networkx_nodes(subG_mod, pos1, ax=ax1, node_size=40, node_color='#5DCAA5')
  nx.draw_networkx_edges(subG_mod, pos1, ax=ax1, alpha=0.4)
  ax1.set_title(f'DSLM-Mod Community\n{len(nodes_mod)} nodes, {subG_mod.number_of_edges()} edges)', fontsize=11)
  ax1.axis('off')

  pos2 = nx.spring_layout(subG_map, seed=42)
  nx.draw_networkx_nodes(subG_map, pos2, ax=ax2, node_size=40, node_color='#AFA9EC')
  nx.draw_networkx_edges(subG_map, pos2, ax=ax2, alpha=0.4)
  ax2.set_title(f'DSLM-Map Community\n{len(nodes_map)} nodes, {subG_map.number_of_edges()} edges)', fontsize=11)
  ax2.axis('off')

  plt.suptitle('Detected Communities: DSLM-Mod vs DSLM-Map', fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.show()

  # --- Also show baseline communities for comparison ---
  comm_nodes_louvain = invert_partition(result_louvain)
  cid_louv, nodes_louv = pick_medium_community(comm_nodes_louvain)
  subG_louv = G.subgraph(nodes_louv).copy()

  comm_nodes_info = invert_partition(result_infomap)
  cid_info, nodes_info = pick_medium_community(comm_nodes_info)
  subG_info = G.subgraph(nodes_info).copy()

  fig, axes = plt.subplots(2, 2, figsize=(14, 12))

  # Louvain
  pos = nx.spring_layout(subG_louv, seed=42)
  nx.draw_networkx_nodes(subG_louv, pos, ax=axes[0, 0], node_size=40, node_color='#85B7EB')
  nx.draw_networkx_edges(subG_louv, pos, ax=axes[0, 0], alpha=0.4)
  axes[0, 0].set_title(f'Louvain (Library)\n({len(nodes_louv)} nodes)', fontsize=11)
  axes[0, 0].axis('off')

  # Infomap
  pos = nx.spring_layout(subG_info, seed=42)
  nx.draw_networkx_nodes(subG_info, pos, ax=axes[0, 1], node_size=40, node_color='#85B7EB')
  nx.draw_networkx_edges(subG_info, pos, ax=axes[0, 1], alpha=0.4)
  axes[0, 1].set_title(f'Infomap (Library)\n({len(nodes_info)} nodes)', fontsize=11)
  axes[0, 1].axis('off')

  # DSLM-Mod
  pos = nx.spring_layout(subG_mod, seed=42)
  nx.draw_networkx_nodes(subG_mod, pos, ax=axes[1, 0], node_size=40, node_color='#5DCAA5')
  nx.draw_networkx_edges(subG_mod, pos, ax=axes[1, 0], alpha=0.4)
  axes[1, 0].set_title(f'DSLM-Mod (Ours)\n({len(nodes_mod)} nodes)', fontsize=11)
  axes[1, 0].axis('off')

  # DSLM-Map
  pos = nx.spring_layout(subG_map, seed=42)
  nx.draw_networkx_nodes(subG_map, pos, ax=axes[1, 1], node_size=40, node_color='#AFA9EC')
  nx.draw_networkx_edges(subG_map, pos, ax=axes[1, 1], alpha=0.4)
  axes[1, 1].set_title(f'DSLM-Map (Ours)\n({len(nodes_map)} nodes)', fontsize=11)
  axes[1, 1].axis('off')

  plt.suptitle('Detected Communities: Library Baselines vs Our DSLM', fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.show()