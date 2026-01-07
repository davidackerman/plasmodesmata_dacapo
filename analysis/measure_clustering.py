# %%
import pickle
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import os

figures_dir = "measurement_results/clustering/figures"
data_dir = "measurement_results/clustering/data"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

dataset = "jrc_22ak351-leaf-3m"
cell_id = 102
cell_data = pickle.load(
    open(
        f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/geodesic_distances/{cell_id}_distribution.pkl",
        "rb",
    )
)
distance_matrix = cell_data["distance_matrix"]

K_max = 100
scores = []
for k in range(2, K_max + 1):
    km = KMedoids(n_clusters=k, metric="precomputed").fit(distance_matrix)
    scores.append(silhouette_score(distance_matrix, km.labels_, metric="precomputed"))

import matplotlib.pyplot as plt

fig_scores, ax_scores = plt.subplots()
ax_scores.plot(scores)
fig_scores_path = os.path.join(figures_dir, "kmedoids_silhouette_scores.png")
fig_scores.savefig(fig_scores_path, dpi=300)
plt.close(fig_scores)


# %%
import numpy as np
import pickle
import trimesh


def mesh_area_from_path(mesh_path: str) -> float:
    # Requires: pip install trimesh
    m = trimesh.load(mesh_path, process=False)
    # If it's a Scene, merge into a single mesh
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.dump()))
    return float(m.area)


def rdf_from_distance_matrix(
    distance_matrix: np.ndarray,
    mesh_area: float,
    r_bins: np.ndarray | None = None,
    r_max_fraction: float = 0.33,
    n_bins: int = 60,
    mask: np.ndarray | None = None,
):
    """
    Compute radial distribution function g(r) on a closed surface
    from a precomputed geodesic distance matrix.

    Parameters
    ----------
    distance_matrix : (N,N) array
        Geodesic distances along the surface. Can include inf/NaN on unreachable pairs.
    mesh_area : float
        Total surface area of the closed mesh (same units as distance_matrix).
    r_bins : array, optional
        Bin edges. If None, they are chosen automatically up to
        r_max_fraction * geodesic_diameter using `n_bins`.
    r_max_fraction : float
        If r_bins None, use this fraction of the geodesic diameter for r_max.
    n_bins : int
        Number of bins if r_bins is None.
    mask : (N,) bool, optional
        If given, restricts vertices to mask==True before forming pair distances.

    Returns
    -------
    r_centers : (B,) array
    g_r : (B,) array
    counts_pairs : (B,) array
        Raw counts of unordered pairs in each bin (upper triangle).
    """
    D = np.array(distance_matrix, copy=False)

    # Optional: restrict to a subset of vertices
    if mask is not None:
        idx = np.flatnonzero(mask)
        D = D[np.ix_(idx, idx)]

    N = D.shape[0]
    # Take only unique unordered pairs (i < j)
    iu, ju = np.triu_indices(N, k=1)

    d = D[iu, ju].astype(np.float64)
    d = d[np.isfinite(d)]
    d = d[d > 0.0]  # remove self/zero

    if d.size == 0:
        raise ValueError("No finite, positive pairwise distances found.")

    # Choose bins if not provided
    if r_bins is None:
        geodesic_diameter = d.max()
        r_max = r_max_fraction * geodesic_diameter
        # Guard against tiny/degenerate meshes
        if r_max <= 0:
            r_max = geodesic_diameter * 0.33
        r_bins = np.linspace(0.0, r_max, n_bins + 1)

    counts_pairs, _ = np.histogram(d, bins=r_bins)

    # Convert pair counts to *per-source average* counts:
    # each unordered pair belongs to two ordered pairs, so expected ordered-per-source
    # counts = (2 / N) * counts_pairs
    counts_per_source = (2.0 / N) * counts_pairs

    # Density of points on the surface (assuming one point per vertex)
    rho = N / mesh_area

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    dr = np.diff(r_bins)

    # Expected neighbors per source in an ideal uniform distribution on a 2D manifold
    expected_per_source = rho * (2.0 * np.pi * r_centers * dr)

    # Radial distribution function
    g_r = counts_per_source / expected_per_source

    return r_centers, g_r, counts_pairs


def _upper_tri(d):  # unordered pairs
    iu, ju = np.triu_indices(d.shape[0], k=1)
    x = d[iu, ju]
    return x[np.isfinite(x) & (x > 0)]


def G_function(distance_matrix, r_bins):
    # Nearest-neighbor CDF (G-function) — CDF of each point’s nearest neighbor distance. Clustering ⇒ G rises fast at small r
    D = np.array(distance_matrix, copy=True)
    np.fill_diagonal(D, np.inf)
    nnd = np.min(D, axis=1)
    nnd = nnd[np.isfinite(nnd)]
    cts, _ = np.histogram(nnd, bins=r_bins)
    G = np.cumsum(cts) / len(nnd)
    return r_bins[1:], G


def ripley_KLH(distance_matrix, area, r_bins):
    d = _upper_tri(distance_matrix)
    N = distance_matrix.shape[0]
    # cumulative pair counts
    pair_cts = np.cumsum(np.histogram(d, bins=r_bins)[0])
    # K via unordered pairs (×2 for ordered)
    K = (area / N**2) * (2 * pair_cts)
    r = r_bins[1:]
    L = np.sqrt(np.maximum(K, 0) / np.pi)
    H = L - r
    return r, K, L, H


def analyze_clusters_at_cutoffs(
    distance_matrix,
    radii=[500, 1000, 2000, 3000, 5000],
    density_thresholds=[5, 10, 15, 20, 25, 30],
):
    """
    Analyze clustering at different local density cutoffs.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Pairwise geodesic distances
    radii : list
        Radius values to consider for local density
    density_thresholds : list
        Minimum number of neighbors within radius to be considered "dense"

    Returns:
    --------
    dict : Results containing cluster counts and sizes for each radius/threshold combo
    """
    from sklearn.cluster import DBSCAN
    from collections import Counter

    results = {}
    N = distance_matrix.shape[0]

    for radius in radii:
        # Neighbor counts depend only on radius; reuse for all density thresholds.
        neighbor_counts = np.sum(distance_matrix <= radius, axis=1) - 1  # subtract self

        for density_thresh in density_thresholds:
            # Identify dense points
            dense_points = np.where(neighbor_counts >= density_thresh)[0]

            if len(dense_points) == 0:
                results[(radius, density_thresh)] = {
                    "n_clusters": 0,
                    "cluster_sizes": [],
                    "n_dense_points": 0,
                    "fraction_dense": 0.0,
                }
                continue

            # Extract distance submatrix for dense points only
            dense_distances = distance_matrix[np.ix_(dense_points, dense_points)]

            # Apply DBSCAN clustering on dense points
            # Use radius as eps, min_samples=2 (need at least 2 points for cluster)
            clustering = DBSCAN(eps=radius, min_samples=2, metric="precomputed")
            cluster_labels = clustering.fit_predict(dense_distances)

            # Count clusters (exclude noise points labeled as -1)
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            # Get cluster sizes
            cluster_sizes = []
            for label in unique_labels:
                if label != -1:  # ignore noise
                    cluster_size = np.sum(cluster_labels == label)
                    cluster_sizes.append(cluster_size)

            results[(radius, density_thresh)] = {
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "n_dense_points": len(dense_points),
                "fraction_dense": len(dense_points) / N,
            }

    return results


# ---------------------------
# Example with your paths
# ---------------------------
datasets = ["jrc_22ak351-leaf-2l", "jrc_22ak351-leaf-3r", "jrc_22ak351-leaf-3m"]
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from dask import delayed, compute
from dask.diagnostics import ProgressBar

# Cache mesh areas to avoid repeated trimesh loads on re-runs.
mesh_area_cache = {}

# Create figure with subplots for all statistics
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Storage for averaged metrics per dataset
dataset_metrics = {}

use_dask = True


def process_cell(pkl_file, r_bins, dataset):
    cell_id = None
    try:
        # Extract cell_id from filename
        cell_id = int(os.path.basename(pkl_file).split("_")[0])

        # Check if corresponding mesh file exists
        mesh_path = f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/{dataset}/cell_fixed/meshes/{cell_id}.ply"
        if not os.path.exists(mesh_path):
            return None

        # Load distance matrix (precomputed geodesic)
        cell_data = pickle.load(open(pkl_file, "rb"))
        distance_matrix = cell_data["distance_matrix"]

        # Load mesh area
        if mesh_path in mesh_area_cache:
            A = mesh_area_cache[mesh_path]
        else:
            A = mesh_area_from_path(mesh_path)
            mesh_area_cache[mesh_path] = A

        # Compute RDF (auto bins up to ~1/3 geodesic diameter)
        r, g, counts = rdf_from_distance_matrix(
            distance_matrix, mesh_area=A, r_bins=r_bins
        )

        # Compute CDF (G)
        r_g, G = G_function(distance_matrix, r_bins=r_bins)

        # Compute Ripley's K, L, H
        r_ripley, K, L, H = ripley_KLH(distance_matrix, A, r_bins)

        # Cluster analysis at different density cutoffs
        cluster_results = analyze_clusters_at_cutoffs(distance_matrix)

        return {
            "cell_id": cell_id,
            "g": g,
            "G": G,
            "K": K,
            "L": L,
            "H": H,
            "cluster_results": cluster_results,
            "r": r,
            "r_g": r_g,
            "r_ripley": r_ripley,
        }
    except Exception as e:
        return {"cell_id": cell_id, "error": str(e)}


for dataset in datasets:
    print(f"Processing dataset: {dataset}")

    # Find all pkl files for this dataset
    pkl_pattern = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/geodesic_distances/*_distribution.pkl"
    pkl_files = glob.glob(pkl_pattern)

    # Storage for this dataset's metrics
    all_g = []
    all_G = []
    all_K = []
    all_L = []
    all_H = []
    all_cluster_results = []
    all_r = None
    all_r_g = None
    all_r_ripley = None

    valid_cells = 0

    # Bin edges reused across metrics for this dataset
    r_bins = np.linspace(0, 20000, 20000 // 500)

    if use_dask:
        tasks = [
            delayed(process_cell)(pkl_file, r_bins, dataset) for pkl_file in pkl_files
        ]
        with ProgressBar():
            results = compute(*tasks, scheduler="threads")
    else:
        results = []
        for pkl_file in tqdm(pkl_files, desc=f"Processing {dataset}", leave=False):
            results.append(process_cell(pkl_file, r_bins, dataset))

    for result in results:
        if not result:
            continue
        if "error" in result:
            print(
                f"Error processing cell {result['cell_id']} in {dataset}: {result['error']}"
            )
            continue

        # Store metrics
        all_g.append(result["g"])
        all_G.append(result["G"])
        all_K.append(result["K"])
        all_L.append(result["L"])
        all_H.append(result["H"])
        all_cluster_results.append(result["cluster_results"])

        # Store r values (should be consistent across cells)
        if all_r is None:
            all_r = result["r"]
            all_r_g = result["r_g"]
            all_r_ripley = result["r_ripley"]

        valid_cells += 1

    if valid_cells > 0:
        print(f"Successfully processed {valid_cells} cells for {dataset}")

        # Compute averages
        avg_g = np.mean(all_g, axis=0)
        avg_G = np.mean(all_G, axis=0)
        avg_K = np.mean(all_K, axis=0)
        avg_L = np.mean(all_L, axis=0)
        avg_H = np.mean(all_H, axis=0)

        # Average cluster results across all cells
        avg_cluster_results = {}
        if all_cluster_results:
            # Get all unique (radius, density_thresh) combinations
            all_keys = set()
            for cell_results in all_cluster_results:
                all_keys.update(cell_results.keys())

            # Average each metric for each combination
            for key in all_keys:
                n_clusters_list = []
                cluster_sizes_list = []
                n_dense_points_list = []
                fraction_dense_list = []

                for cell_results in all_cluster_results:
                    if key in cell_results:
                        n_clusters_list.append(cell_results[key]["n_clusters"])
                        cluster_sizes_list.extend(cell_results[key]["cluster_sizes"])
                        n_dense_points_list.append(cell_results[key]["n_dense_points"])
                        fraction_dense_list.append(cell_results[key]["fraction_dense"])

                avg_cluster_results[key] = {
                    "avg_n_clusters": (
                        np.mean(n_clusters_list) if n_clusters_list else 0
                    ),
                    "std_n_clusters": np.std(n_clusters_list) if n_clusters_list else 0,
                    "all_cluster_sizes": cluster_sizes_list,
                    "avg_cluster_size": (
                        np.mean(cluster_sizes_list) if cluster_sizes_list else 0
                    ),
                    "avg_n_dense_points": (
                        np.mean(n_dense_points_list) if n_dense_points_list else 0
                    ),
                    "avg_fraction_dense": (
                        np.mean(fraction_dense_list) if fraction_dense_list else 0
                    ),
                    "n_cells_contributing": len(n_clusters_list),
                }

        # Store for plotting
        dataset_metrics[dataset] = {
            "r": all_r,
            "g": avg_g,
            "r_g": all_r_g,
            "G": avg_G,
            "r_ripley": all_r_ripley,
            "K": avg_K,
            "L": avg_L,
            "H": avg_H,
            "cluster_analysis": avg_cluster_results,
            "n_cells": valid_cells,
        }

        # Plot averaged metrics
        # Plot RDF
        axes[0, 0].plot(
            all_r, avg_g, marker="o", lw=1, label=f"{dataset} (n={valid_cells})"
        )

        # Plot CDF (G)
        axes[0, 1].plot(
            all_r_g, avg_G, marker="o", lw=1, label=f"{dataset} (n={valid_cells})"
        )

        # Plot Ripley's K
        axes[0, 2].plot(
            all_r_ripley, avg_K, marker="o", lw=1, label=f"{dataset} (n={valid_cells})"
        )

        # Plot Ripley's L
        axes[1, 0].plot(
            all_r_ripley, avg_L, marker="o", lw=1, label=f"{dataset} (n={valid_cells})"
        )

        # Plot Ripley's H
        axes[1, 1].plot(
            all_r_ripley, avg_H, marker="o", lw=1, label=f"{dataset} (n={valid_cells})"
        )
    else:
        print(f"No valid cells found for {dataset}")

# Finalize RDF plot
axes[0, 0].set_xlabel("geodesic distance r")
axes[0, 0].set_ylabel("g(r)")
axes[0, 0].set_title(f"RDF")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Finalize CDF plot
axes[0, 1].set_xlabel("geodesic distance r")
axes[0, 1].set_ylabel("G(r)")
axes[0, 1].set_title(f"CDF (G)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Finalize Ripley's K plot
axes[0, 2].set_xlabel("geodesic distance r")
axes[0, 2].set_ylabel("K(r)")
axes[0, 2].set_title(f"Ripley's K")
axes[0, 2].set_xscale("log")  # Set x-axis to logarithmic scale base 2
axes[0, 2].set_yscale("log")  # Set y-axis to logarithmic scale base 2
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
# Add reference line y=π*r² for random distribution
x_lim = axes[0, 2].get_xlim()
r_ref = np.linspace(x_lim[0], x_lim[1], 100)
k_random = np.pi * r_ref**2
axes[0, 2].plot(r_ref, k_random, "k--", alpha=0.5, label="Random (K=πr²)")

# Finalize Ripley's L plot
axes[1, 0].set_xlabel("geodesic distance r")
axes[1, 0].set_ylabel("L(r)")
axes[1, 0].set_title(f"Ripley's L")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
# Add reference line y=x for random distribution
x_lim = axes[1, 0].get_xlim()
axes[1, 0].plot(x_lim, x_lim, "k--", alpha=0.5, label="Random (L=r)")

# Finalize Ripley's H plot
axes[1, 1].set_xlabel("geodesic distance r")
axes[1, 1].set_ylabel("H(r)")
axes[1, 1].set_title(f"Ripley's H")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(
    y=0, color="black", linestyle="--", alpha=0.5
)  # Reference line at H=0

# Hide the empty subplot
axes[1, 2].axis("off")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "measure_clustering_summary.png")
plt.savefig(fig_path, dpi=300)
plt.close(fig)

# Print cluster analysis summary
print("\n" + "=" * 80)
print("CLUSTER ANALYSIS SUMMARY")
print("=" * 80)

for dataset in datasets:
    if dataset in dataset_metrics and "cluster_analysis" in dataset_metrics[dataset]:
        summary_lines = []
        print(f"\n{dataset.upper()}:")
        summary_lines.append(f"{dataset.upper()}:")
        print("-" * 50)
        summary_lines.append("-" * 50)

        cluster_data = dataset_metrics[dataset]["cluster_analysis"]

        # Print results organized by radius
        radii = sorted(set(key[0] for key in cluster_data.keys()))
        density_thresholds = sorted(set(key[1] for key in cluster_data.keys()))

        print(
            f"{'Radius':<8} {'DensThresh':<12} {'AvgClusters':<12} {'StdClusters':<12} {'AvgClusterSize':<15} {'FracDense':<10}"
        )
        summary_lines.append(
            f"{'Radius':<8} {'DensThresh':<12} {'AvgClusters':<12} {'StdClusters':<12} {'AvgClusterSize':<15} {'FracDense':<10}"
        )
        print("-" * 80)
        summary_lines.append("-" * 80)

        for radius in radii:
            for density_thresh in density_thresholds:
                key = (radius, density_thresh)
                if key in cluster_data:
                    data = cluster_data[key]
                    line = (
                        f"{radius:<8} {density_thresh:<12} {data['avg_n_clusters']:<12.2f} "
                        f"{data['std_n_clusters']:<12.2f} {data['avg_cluster_size']:<15.2f} "
                        f"{data['avg_fraction_dense']:<10.3f}"
                    )
                    print(line)
                    summary_lines.append(line)

        # Summary statistics
        print(f"\nSummary for {dataset}:")
        summary_lines.append("")
        summary_lines.append(f"Summary for {dataset}:")
        max_clusters_key = max(
            cluster_data.keys(), key=lambda k: cluster_data[k]["avg_n_clusters"]
        )
        max_clusters_data = cluster_data[max_clusters_key]
        line = f"  Max avg clusters: {max_clusters_data['avg_n_clusters']:.2f} at radius={max_clusters_key[0]}, density_thresh={max_clusters_key[1]}"
        print(line)
        summary_lines.append(line)

        # Find optimal clustering parameters (high cluster count, reasonable cluster size)
        optimal_configs = [
            (k, v)
            for k, v in cluster_data.items()
            if v["avg_n_clusters"] >= 2 and 3 <= v["avg_cluster_size"] <= 20
        ]
        if optimal_configs:
            optimal_key, optimal_data = max(
                optimal_configs, key=lambda x: x[1]["avg_n_clusters"]
            )
            line = (
                f"  Optimal config: radius={optimal_key[0]}, density_thresh={optimal_key[1]} "
                f"-> {optimal_data['avg_n_clusters']:.2f} clusters, {optimal_data['avg_cluster_size']:.2f} avg size"
            )
            print(line)
            summary_lines.append(line)

        summary_path = os.path.join(data_dir, f"{dataset}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
    else:
        summary_path = os.path.join(data_dir, f"{dataset}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"{dataset.upper()}: No cluster analysis data available.")

print("\n" + "=" * 80)

# Create heatmaps for cluster analysis
print("Creating cluster analysis heatmaps...")

# Extract unique radii and density thresholds for heatmap axes
all_radii = set()
all_density_thresholds = set()
for dataset in datasets:
    if dataset in dataset_metrics and "cluster_analysis" in dataset_metrics[dataset]:
        cluster_data = dataset_metrics[dataset]["cluster_analysis"]
        for radius, density_thresh in cluster_data.keys():
            all_radii.add(radius)
            all_density_thresholds.add(density_thresh)

radii_sorted = sorted(all_radii)
density_thresholds_sorted = sorted(all_density_thresholds)

if radii_sorted and density_thresholds_sorted:
    # Create figure with 2 rows (cluster count, cluster size) and 3 columns (datasets)
    fig_heatmaps, axes_heatmaps = plt.subplots(2, 3, figsize=(18, 12))

    # First pass: collect all data to determine global min/max for consistent color scaling
    all_cluster_counts = []
    all_cluster_sizes = []
    all_matrices = {}

    for dataset in datasets:
        if (
            dataset in dataset_metrics
            and "cluster_analysis" in dataset_metrics[dataset]
        ):
            cluster_data = dataset_metrics[dataset]["cluster_analysis"]

            # Initialize matrices for heatmaps
            cluster_count_matrix = np.full(
                (len(density_thresholds_sorted), len(radii_sorted)), np.nan
            )
            cluster_size_matrix = np.full(
                (len(density_thresholds_sorted), len(radii_sorted)), np.nan
            )

            # Fill matrices with data
            for i, density_thresh in enumerate(density_thresholds_sorted):
                for j, radius in enumerate(radii_sorted):
                    key = (radius, density_thresh)
                    if key in cluster_data:
                        cluster_count_matrix[i, j] = cluster_data[key]["avg_n_clusters"]
                        cluster_size_matrix[i, j] = cluster_data[key][
                            "avg_cluster_size"
                        ]

            # Store matrices and collect values for global scaling
            all_matrices[dataset] = (cluster_count_matrix, cluster_size_matrix)
            all_cluster_counts.extend(
                cluster_count_matrix[~np.isnan(cluster_count_matrix)]
            )
            all_cluster_sizes.extend(
                cluster_size_matrix[~np.isnan(cluster_size_matrix)]
            )

    # Calculate global min/max for consistent color scaling
    if all_cluster_counts:
        count_vmin, count_vmax = min(all_cluster_counts), max(all_cluster_counts)
    else:
        count_vmin, count_vmax = 0, 1

    if all_cluster_sizes:
        size_vmin, size_vmax = min(all_cluster_sizes), max(all_cluster_sizes)
    else:
        size_vmin, size_vmax = 0, 1

    for col, dataset in enumerate(datasets):
        if (
            dataset in dataset_metrics
            and "cluster_analysis" in dataset_metrics[dataset]
        ):
            # Get pre-calculated matrices
            cluster_count_matrix, cluster_size_matrix = all_matrices[dataset]

            # Plot cluster count heatmap (top row) with global color scale
            # Flip matrix vertically so smallest density threshold is at bottom
            im1 = axes_heatmaps[0, col].imshow(
                np.flipud(cluster_count_matrix),
                cmap="viridis",
                aspect="auto",
                interpolation="nearest",
                vmin=count_vmin,
                vmax=count_vmax,
            )
            axes_heatmaps[0, col].set_title(f"{dataset}\nAvg Number of Clusters")
            axes_heatmaps[0, col].set_xlabel("Radius")
            axes_heatmaps[0, col].set_ylabel("Density Threshold")

            # Set ticks and labels
            axes_heatmaps[0, col].set_xticks(range(len(radii_sorted)))
            axes_heatmaps[0, col].set_xticklabels(
                [str(r) for r in radii_sorted], rotation=45
            )
            axes_heatmaps[0, col].set_yticks(range(len(density_thresholds_sorted)))
            # Reverse the order of y-tick labels so smallest is at bottom
            axes_heatmaps[0, col].set_yticklabels(
                [str(d) for d in reversed(density_thresholds_sorted)]
            )

            # Add colorbar
            plt.colorbar(im1, ax=axes_heatmaps[0, col], shrink=0.8)

            # Plot cluster size heatmap (bottom row) with global color scale
            # Flip matrix vertically so smallest density threshold is at bottom
            im2 = axes_heatmaps[1, col].imshow(
                np.flipud(cluster_size_matrix),
                cmap="viridis",
                aspect="auto",
                interpolation="nearest",
                vmin=size_vmin,
                vmax=size_vmax,
            )
            axes_heatmaps[1, col].set_title(f"{dataset}\nAvg Cluster Size")
            axes_heatmaps[1, col].set_xlabel("Radius")
            axes_heatmaps[1, col].set_ylabel("Density Threshold")

            # Set ticks and labels
            axes_heatmaps[1, col].set_xticks(range(len(radii_sorted)))
            axes_heatmaps[1, col].set_xticklabels(
                [str(r) for r in radii_sorted], rotation=45
            )
            axes_heatmaps[1, col].set_yticks(range(len(density_thresholds_sorted)))
            # Reverse the order of y-tick labels so smallest is at bottom
            axes_heatmaps[1, col].set_yticklabels(
                [str(d) for d in reversed(density_thresholds_sorted)]
            )

            # Add colorbar
            plt.colorbar(im2, ax=axes_heatmaps[1, col], shrink=0.8)

            # Add value annotations on heatmaps with larger font
            # Note: since we flipped the matrix, we need to adjust the y-coordinate
            for i in range(len(density_thresholds_sorted)):
                for j in range(len(radii_sorted)):
                    # For flipped matrix, use (len - 1 - i) for y-coordinate
                    flipped_i = len(density_thresholds_sorted) - 1 - i
                    if not np.isnan(cluster_count_matrix[i, j]):
                        axes_heatmaps[0, col].text(
                            j,
                            flipped_i,
                            f"{cluster_count_matrix[i, j]:.1f}",
                            ha="center",
                            va="center",
                            color=(
                                "white"
                                if cluster_count_matrix[i, j]
                                > (count_vmin + count_vmax) / 2
                                else "black"
                            ),
                            fontsize=12,
                            fontweight="bold",
                        )
                    if not np.isnan(cluster_size_matrix[i, j]):
                        axes_heatmaps[1, col].text(
                            j,
                            flipped_i,
                            f"{cluster_size_matrix[i, j]:.1f}",
                            ha="center",
                            va="center",
                            color=(
                                "white"
                                if cluster_size_matrix[i, j]
                                > (size_vmin + size_vmax) / 2
                                else "black"
                            ),
                            fontsize=12,
                            fontweight="bold",
                        )
        else:
            # If no data available, hide the subplots
            axes_heatmaps[0, col].axis("off")
            axes_heatmaps[1, col].axis("off")
            axes_heatmaps[0, col].text(
                0.5,
                0.5,
                f"No data\nfor {dataset}",
                ha="center",
                va="center",
                transform=axes_heatmaps[0, col].transAxes,
            )

    plt.tight_layout()
    heatmap_path = os.path.join(figures_dir, "cluster_analysis_heatmaps.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close(fig_heatmaps)

    print("Heatmaps created successfully!")
else:
    print("No cluster analysis data available for heatmaps.")

# %%
# from funlib.persistence import open_ds
# from funlib.geometry import Roi
# import numpy as np
# import imageio


# def write_mp4_lossless_gray(frames_3d, path, fps=30):
#     """
#     frames_3d: (T, H, W), any dtype -> uint8
#     path: e.g. './tmp.mp4'
#     """
#     with imageio.get_writer(
#         path,
#         format="ffmpeg",                 # <-- force ffmpeg
#         fps=fps,
#         codec="libx264",                 # H.264
#         ffmpeg_params=["-crf","0","-preset","veryslow","-pix_fmt","gray"],  # lossless
#     ) as w:
#         for f in frames_3d:
#             w.append_data(np.ascontiguousarray(f))


# ds = open_ds(
#     "/groups/funceworm/funceworm/adult/Adult_Day1_DatasetA4/jrc_P3_E5_D1_N2_trimmed_align_v2.zarr",
#     "s0",
# )
# data = ds.to_ndarray(Roi(np.array((1354, 1000, 6053)) * 32, [400 * 8] * 3))
# write_mp4_lossless_gray(data, "./tmp.mp4")
# %%
