import os
import networkx as nx
import numpy as np
import csv
from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra  # Import GlobalHydra explicitly
from topobench.utils.config_resolvers import (
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    infer_in_channels,
)

# Clear GlobalHydra instance if already initialized
if GlobalHydra().is_initialized():
    GlobalHydra().clear()

initialize(config_path="../configs", job_name="job")

def load_dataset(adj_metric="wgcna", adj_thresh=0.5):
    """
    Load the FTD dataset with a specified adjacency threshold.
    """
    cfg = compose(
        config_name="run.yaml",
        overrides=[
            "model=graph/gat",
            "dataset=graph/FTD",
            f"dataset.loader.parameters.adj_metric={adj_metric}",
            f"dataset.loader.parameters.adj_thresh={adj_thresh}",
        ], 
        return_hydra_config=True
    )
    loader = instantiate(cfg.dataset.loader)
    _, _ = loader.load()
    return loader.datasets[0]

def get_graph_stats(ftd_dataset):
    """
    Get statistics of the graph.
    """
    # Load the adjacency matrix
    adj_matrix = ftd_dataset.get_adjacency_matrix(ftd_dataset.adj_path, ftd_dataset.config.adj_thresh, ftd_dataset.config)
    # Generate a graph from the adjacency matrix
    graph = nx.from_numpy_matrix(adj_matrix.cpu().numpy())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    # Calculate statistics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = np.mean([d for n, d in graph.degree()])
    density = nx.density(graph)
    number_connected_components = nx.number_connected_components(graph)
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
        "number_connected_components": number_connected_components,
    }
    
# from concurrent.futures import ThreadPoolExecutor

# def process_adj_thresh(adj_thresh, output_file, fieldnames):
#     """
#     Process a single adjacency threshold and write the stats to the CSV file.
#     """
#     print(f"Processing adj_thresh={adj_thresh}...")
#     # Load the dataset
#     ftd_dataset = load_dataset(adj_thresh=adj_thresh)
#     # Get graph statistics
#     stats = get_graph_stats(ftd_dataset)
#     # Add the adjacency threshold to the stats
#     stats["adj_thresh"] = adj_thresh

#     # Append the stats to the CSV file
#     with open(output_file, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writerow(stats)  # Write the current row

#     print(f"Saved stats for adj_thresh={adj_thresh} to {output_file}")
    
if __name__ == "__main__":
    # Define the output file and fieldnames
    adj_metric = "spearman_correlation"  # Change this to the desired adjacency metric
    directory = "./tutorials/stats/"+adj_metric+"/"
    output_file = directory + "/graph_stats.csv"
    fieldnames = ["adj_thresh", "num_nodes", "num_edges", "avg_degree", "density", "number_connected_components"]

    # os.makedirs(os.path.dirname(directory), exist_ok=True)
    # # Open the file in write mode initially to write the header
    # with open(output_file, "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()  # Write the header row
    
    # Create a ThreadPoolExecutor to parallelize the computation
    # with ThreadPoolExecutor(max_workers=100) as executor:
    #     # Generate a list of adjacency thresholds
    #     adj_thresh_list = [idx / 100.0 for idx in range(91, -1, -1)]
    #     # Submit tasks to the executor
    #     futures = [
    #         executor.submit(process_adj_thresh, adj_thresh, output_file, fieldnames)
    #         for adj_thresh in adj_thresh_list
    #     ]

    #     # Wait for all threads to complete
    #     for future in futures:
    #         future.result()

    # print(f"All computations completed. Results saved to {output_file}")

    for idx in range(23, -1, -1):
        adj_thresh = idx / 100.0
        print(f"Processing adj_metric={adj_metric} with adj_thresh={adj_thresh}...")
        # Load the dataset
        ftd_dataset = load_dataset(adj_metric=adj_metric, adj_thresh=adj_thresh)
        # Get graph statistics
        stats = get_graph_stats(ftd_dataset)
        # Add the adjacency threshold to the stats
        stats["adj_thresh"] = adj_thresh

        # Append the stats to the CSV file
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(stats)  # Write the current row

        print(f"Saved stats for adj_thresh={adj_thresh} to {output_file}")