#!/usr/bin/env python3
"""
Fix the interactive HTML visualizations by regenerating them with proper NumPy array serialization.
"""
import pickle
import os
import numpy as np
import plotly.graph_objects as go
from sklearn_extra.cluster import KMedoids
import sys

# Import the fixed function
from cluster_cells_by_pattern import create_interactive_pca_with_projections

def main():
    # Load existing results
    cache_file = "measurement_results/cell_clustering/data/cell_clustering_results.pkl"
    print("Loading clustering results...")

    if not os.path.exists(cache_file):
        print(f"Error: Cache file not found at {cache_file}")
        sys.exit(1)

    with open(cache_file, 'rb') as f:
        results_dict = pickle.load(f)

    # Extract variables
    cell_ids = results_dict["cell_ids"]
    cell_datasets = results_dict["datasets"]
    X_scaled = results_dict["X_scaled"]
    X_pca = results_dict["X_pca"]
    all_results = results_dict["all_results"]

    # Cluster with k=2
    print("Clustering with k=2...")
    km = KMedoids(n_clusters=2, metric='euclidean', random_state=42)
    labels_k2 = km.fit_predict(X_scaled)

    # Get unique datasets
    datasets = sorted(set(cell_datasets))
    print(f"Found {len(datasets)} datasets: {datasets}")

    # Generate per-dataset visualizations for k=2
    figures_dir = "measurement_results/cell_clustering/figures/k2"

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Filter to cells from this dataset
        dataset_mask = np.array([ds == dataset for ds in cell_datasets])
        dataset_indices = np.where(dataset_mask)[0]

        if len(dataset_indices) == 0:
            print(f"  No cells found for {dataset}, skipping")
            continue

        print(f"  Found {len(dataset_indices)} cells")

        dataset_X_pca = X_pca[dataset_mask]
        dataset_cluster_labels = labels_k2[dataset_mask]
        dataset_cell_ids = [cell_ids[i] for i in dataset_indices]
        dataset_cell_datasets = [cell_datasets[i] for i in dataset_indices]
        dataset_results = [all_results[i] for i in dataset_indices]

        # Create dataset-specific subdirectory
        dataset_dir = os.path.join(figures_dir, dataset.split("-")[-1])
        os.makedirs(dataset_dir, exist_ok=True)

        # Create interactive visualization for this dataset
        print(f"  Creating visualization...")
        create_interactive_pca_with_projections(
            dataset_X_pca,
            dataset_cluster_labels,
            dataset_cell_ids,
            dataset_cell_datasets,
            dataset_results,
            [dataset],
            2,  # k_val
            dataset_dir,
        )
        print(f"  Done! Saved to {dataset_dir}/interactive_pca_mollweide_k2.html")

    print("\nAll HTML files regenerated successfully!")

if __name__ == "__main__":
    main()
