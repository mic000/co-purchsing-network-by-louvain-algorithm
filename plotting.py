

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