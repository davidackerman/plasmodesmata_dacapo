"""
Cell-level clustering by spatial pattern signature.

This script clusters individual cells based on their plasmodesmata spatial patterns,
using features extracted from:
- Ripley's H function (H_peak, r_H_peak, H_area_pos, r_zero_cross)
- Radial distribution function g(r) (g_peak, r_g_peak, g_small_mean)
- Nearest-neighbor CDF G(r) (quantiles: G_r10, G_r50, G_r90)
- DBSCAN sweep results (nclusters_max, radius_at_max, density_thresh_at_max, frac_dense_at_max)

The goal is to identify "texture regimes" - groups of cells with similar clustering patterns
(e.g., few large patches vs many small clusters vs uniform distribution).
"""

import pickle
import os
import numpy as np
import glob
from tqdm import tqdm
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import trimesh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json

# Import functions from measure_clustering
from measure_clustering import (
    mesh_area_from_path,
    rdf_from_distance_matrix,
    G_function,
    ripley_KLH,
    analyze_clusters_at_cutoffs,
)

# ===== CONFIGURATION =====
# Set to True to use existing processed results and skip recomputation
USE_EXISTING_DATA = False
# =========================

# Output directories
figures_dir = "measurement_results/cell_clustering/figures"
data_dir = "measurement_results/cell_clustering/data"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


def extract_features_from_curves(result):
    """
    Extract interpretable scalar features from per-cell spatial statistics.

    Parameters
    ----------
    result : dict
        Output from process_cell() containing r, g, G, H, and cluster_results

    Returns
    -------
    dict
        Dictionary of feature name -> value
    """
    features = {}

    # --- Ripley's H features ---
    H = result["H"]
    r = result["r_ripley"]

    # Peak height and location
    features["H_peak"] = np.max(H)
    features["r_H_peak"] = r[np.argmax(H)]

    # Integrated positive H (area of clustering signal)
    features["H_area_pos"] = np.trapz(np.maximum(H, 0), r)

    # Zero crossing (scale of clustering transition)
    zero_crossings = np.where(np.diff(np.sign(H)))[0]
    if len(zero_crossings) > 0:
        features["r_zero_cross"] = r[zero_crossings[0]]
    else:
        features["r_zero_cross"] = r[-1]  # No crossing = strong clustering throughout

    # H variance (captures complexity of pattern)
    features["H_variance"] = np.var(H)

    # --- g(r) features ---
    g = result["g"]
    r_g = result["r"]

    # Peak height and location
    features["g_peak"] = np.max(g)
    features["r_g_peak"] = r_g[np.argmax(g)]

    # Mean g at small scales (< 2000 nm)
    small_r_mask = r_g < 2000
    if np.any(small_r_mask):
        features["g_small_mean"] = np.mean(g[small_r_mask])
    else:
        features["g_small_mean"] = g[0]

    # g decay rate (how fast clustering weakens)
    # Fit exponential decay to g(r) after peak
    peak_idx = np.argmax(g)
    if peak_idx < len(g) - 1:
        g_after_peak = g[peak_idx:]
        r_after_peak = r_g[peak_idx:]
        # Simple decay measure: slope in log space
        if np.all(g_after_peak > 0):
            log_g = np.log(g_after_peak)
            features["g_decay_rate"] = -np.polyfit(r_after_peak, log_g, 1)[0]
        else:
            features["g_decay_rate"] = 0
    else:
        features["g_decay_rate"] = 0

    # --- G(r) features (nearest neighbor distribution) ---
    G = result["G"]
    r_G = result["r_g"]

    # Quantiles of nearest neighbor distances
    # Interpolate to find radii at specific CDF values
    features["G_r10"] = np.interp(0.1, G, r_G)
    features["G_r50"] = np.interp(0.5, G, r_G)
    features["G_r90"] = np.interp(0.9, G, r_G)

    # Median nearest neighbor distance
    features["median_nn_dist"] = features["G_r50"]

    # --- DBSCAN sweep features (clustering regime) ---
    cluster_results = result["cluster_results"]

    # Find parameter combination that maximizes number of clusters
    max_clusters = 0
    best_key = None
    for key, data in cluster_results.items():
        if data["n_clusters"] > max_clusters:
            max_clusters = data["n_clusters"]
            best_key = key

    if best_key is not None:
        best_data = cluster_results[best_key]
        features["nclusters_max"] = best_data["n_clusters"]
        features["radius_at_max"] = best_key[0]
        features["density_thresh_at_max"] = best_key[1]
        features["frac_dense_at_max"] = best_data["fraction_dense"]

        # Average cluster size at optimal parameters
        if len(best_data["cluster_sizes"]) > 0:
            features["avg_cluster_size"] = np.mean(best_data["cluster_sizes"])
            features["std_cluster_size"] = np.std(best_data["cluster_sizes"])

            # Effective number of clusters (entropy-based, captures "2 big vs 100 small")
            sizes = np.array(best_data["cluster_sizes"])
            probs = sizes / sizes.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            features["n_clusters_effective"] = np.exp(entropy)
        else:
            features["avg_cluster_size"] = 0
            features["std_cluster_size"] = 0
            features["n_clusters_effective"] = 0
    else:
        # No clusters found at any parameter setting
        features["nclusters_max"] = 0
        features["radius_at_max"] = 0
        features["density_thresh_at_max"] = 0
        features["frac_dense_at_max"] = 0
        features["avg_cluster_size"] = 0
        features["std_cluster_size"] = 0
        features["n_clusters_effective"] = 0

    return features


def process_cell(pkl_file, r_bins, dataset):
    """
    Process a single cell and extract both curves and features.

    This is adapted from measure_clustering.py to also extract features.
    """
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
        A = mesh_area_from_path(mesh_path)

        # Compute RDF
        r, g, counts = rdf_from_distance_matrix(
            distance_matrix, mesh_area=A, r_bins=r_bins
        )

        # Compute CDF (G)
        r_g, G = G_function(distance_matrix, r_bins=r_bins)

        # Compute Ripley's K, L, H
        r_ripley, K, L, H = ripley_KLH(distance_matrix, A, r_bins)

        # Cluster analysis at different density cutoffs
        cluster_results = analyze_clusters_at_cutoffs(distance_matrix)

        result = {
            "cell_id": cell_id,
            "dataset": dataset,
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

        # Extract features
        features = extract_features_from_curves(result)
        result["features"] = features

        return result

    except Exception as e:
        return {"cell_id": cell_id, "dataset": dataset, "error": str(e)}


def create_mollweide_projection(mesh_vertices, plasmodesmata_positions, title=""):
    """
    Create a mollweide projection of plasmodesmata on a mesh surface.

    Parameters
    ----------
    mesh_vertices : (N, 3) array
        Mesh vertex positions
    plasmodesmata_positions : (M, 3) array
        Positions of plasmodesmata on the mesh surface
    title : str
        Title for the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with mollweide projection
    """
    # Calculate center of mass
    com = np.mean(mesh_vertices, axis=0)

    # Center plasmodesmata positions around COM
    centered_pd = plasmodesmata_positions - com

    # Convert to spherical coordinates (theta, phi)
    # theta: azimuthal angle [0, 2π]
    # phi: polar angle from z-axis [0, π]
    r = np.linalg.norm(centered_pd, axis=1)
    theta = np.arctan2(centered_pd[:, 1], centered_pd[:, 0])  # azimuth
    phi = np.arccos(np.clip(centered_pd[:, 2] / (r + 1e-10), -1, 1))  # polar

    # Convert to longitude/latitude for mollweide
    # longitude: [-π, π], latitude: [-π/2, π/2]
    lon = theta
    lat = np.pi / 2 - phi

    # Create mollweide projection
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="mollweide")

    # Plot plasmodesmata
    ax.scatter(lon, lat, s=2, alpha=0.6, c="red")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Remove degree labels on axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return fig


def _generate_single_projection(args):
    """
    Helper function to generate a single mollweide projection.
    Used for parallel processing.
    """
    idx, result, cluster_label, X_pca_row, projections_dir = args

    cell_id = result["cell_id"]
    dataset = result["dataset"]

    try:
        mesh_path = f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/{dataset}/cell_fixed/meshes/{cell_id}.ply"
        pkl_path = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/geodesic_distances/{cell_id}_distribution.pkl"

        if not os.path.exists(mesh_path) or not os.path.exists(pkl_path):
            return None

        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.dump()))

        cell_data = pickle.load(open(pkl_path, "rb"))

        # Get plasmodesmata positions
        pd_indices = cell_data["plasmodesmata_indices"]
        updated_vertices = cell_data["updated_vertices"]
        pd_positions = updated_vertices[pd_indices]

        # Create mollweide projection
        fig = create_mollweide_projection(
            mesh.vertices,
            pd_positions,
            title=f"Cell {cell_id} ({dataset.split('-')[-1]}) - Cluster {cluster_label}",
        )

        # Save as PNG file using cell_id in filename
        png_filename = f"mollweide_{cell_id}.png"
        png_path = os.path.join(projections_dir, png_filename)
        fig.savefig(png_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        return {
            "idx": int(idx),
            "img_path": png_filename,
            "cell_id": int(cell_id),
            "dataset": dataset,
            "cluster": int(cluster_label),
            "pc1": float(X_pca_row[0]),
            "pc2": float(X_pca_row[1]),
        }

    except Exception as e:
        print(f"  Warning: Failed to create projection for cell {cell_id}: {e}")
        return None


def create_interactive_pca_with_projections(
    X_pca,
    cluster_labels,
    cell_ids,
    cell_datasets,
    all_results,
    datasets,
    k_val,
    figures_dir,
):
    """
    Create an interactive HTML visualization where hovering over PCA points shows
    the corresponding cell's mollweide projection.

    Parameters
    ----------
    X_pca : array
        PCA-transformed features
    cluster_labels : array
        Cluster assignments
    cell_ids : list
        Cell IDs
    cell_datasets : list
        Dataset names for each cell
    all_results : list
        Full results containing cell data
    datasets : list
        Unique dataset names
    k_val : int
        Number of clusters
    figures_dir : str
        Directory to save figures
    """
    print(f"  Creating interactive PCA visualization with mollweide projections...")

    # Create subdirectory for projection images
    projections_dir = os.path.join(figures_dir, "projections")
    os.makedirs(projections_dir, exist_ok=True)

    # Generate mollweide projections for ALL cells in parallel
    print(
        f"  Generating mollweide projections for all {len(all_results)} cells in parallel..."
    )

    # Prepare arguments for parallel processing
    tasks = [
        (idx, all_results[idx], cluster_labels[idx], X_pca[idx], projections_dir)
        for idx in range(len(all_results))
    ]

    # Use Dask for parallel processing
    projection_tasks = [delayed(_generate_single_projection)(task) for task in tasks]

    with ProgressBar():
        projection_results = compute(*projection_tasks, scheduler="threads")

    # Filter out failed projections and build projection_images dict
    projection_images = {}
    for result in projection_results:
        if result is not None:
            idx = result.pop("idx")
            projection_images[idx] = result

    # Create interactive Plotly figure with hover images
    hover_text = []
    for i in range(len(cell_ids)):
        hover_text.append(
            f"Cell {cell_ids[i]}<br>"
            f"Dataset: {cell_datasets[i].split('-')[-1]}<br>"
            f"Cluster: {cluster_labels[i]}<br>"
            f"PC1: {X_pca[i, 0]:.2f}<br>"
            f"PC2: {X_pca[i, 1]:.2f}"
        )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=X_pca[:, 0].tolist(),
                y=X_pca[:, 1].tolist(),
                mode="markers",
                marker=dict(
                    size=8,
                    color=cluster_labels.tolist() if hasattr(cluster_labels, 'tolist') else list(cluster_labels),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Cluster"),
                    line=dict(width=1, color="DarkSlateGray"),
                ),
                text=hover_text,
                hovertemplate="<b>%{text}</b><extra></extra>",
                customdata=list(range(len(cell_ids))),  # Store index for click events
            )
        ]
    )

    fig.update_layout(
        title=f"Interactive PCA with Mollweide Projections (k={k_val})<br>"
        f"<sub>Hover over points to view mollweide projection</sub>",
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=700,
        height=700,
        hovermode="closest",
    )

    # Convert Plotly figure to JSON
    fig_json = fig.to_json()

    # Create HTML with plotly figure and image display area
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive PCA with Mollweide Projections</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 20px;
            }}
            #container {{
                display: flex;
                gap: 20px;
                max-width: 1600px;
                margin: 0 auto;
            }}
            #plotly-div {{
                flex: 1;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            #projection-display {{
                flex: 1;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }}
            #projection-img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .instruction {{
                color: #666;
                font-style: italic;
                margin-bottom: 10px;
                font-size: 14px;
            }}
            #projection-info {{
                margin-top: 15px;
                font-size: 16px;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <h1>Interactive PCA Visualization with Mollweide Projections (k={k_val})</h1>
        <div id="container">
            <div id="plotly-div"></div>
            <div id="projection-display">
                <p class="instruction">Hover over any point in the PCA plot to view its mollweide projection</p>
                <img id="projection-img" src="" alt="Mollweide projection will appear here" style="display:none;">
                <p id="projection-info"></p>
            </div>
        </div>

        <script>
            var projectionData = {json.dumps(projection_images)};

            // Create the plotly figure
            var figData = {fig_json};
            Plotly.newPlot('plotly-div', figData.data, figData.layout);

            // Add hover event listener
            var plotlyDiv = document.getElementById('plotly-div');
            plotlyDiv.on('plotly_hover', function(data) {{
                var pointIndex = data.points[0].customdata;

                if (projectionData.hasOwnProperty(pointIndex)) {{
                    var projInfo = projectionData[pointIndex];
                    var img = document.getElementById('projection-img');
                    var info = document.getElementById('projection-info');

                    img.src = 'projections/' + projInfo.img_path;
                    img.style.display = 'block';
                    info.innerHTML = '<b>Cell ' + projInfo.cell_id + '</b> from dataset ' +
                                    projInfo.dataset.split('-').pop() +
                                    ' (Cluster ' + projInfo.cluster + ')';
                }} else {{
                    var info = document.getElementById('projection-info');
                    info.innerHTML = '<i>No projection available for this cell.</i>';
                }}
            }});

            // Optional: Also support click for "pinning" a projection
            plotlyDiv.on('plotly_click', function(data) {{
                var pointIndex = data.points[0].customdata;

                if (projectionData.hasOwnProperty(pointIndex)) {{
                    var projInfo = projectionData[pointIndex];
                    var img = document.getElementById('projection-img');
                    var info = document.getElementById('projection-info');

                    img.src = 'projections/' + projInfo.img_path;
                    img.style.display = 'block';
                    info.innerHTML = '<b>Cell ' + projInfo.cell_id + '</b> from dataset ' +
                                    projInfo.dataset.split('-').pop() +
                                    ' (Cluster ' + projInfo.cluster + ') <i>(pinned)</i>';
                }}
            }});
        </script>
    </body>
    </html>
    """

    # Save HTML file
    html_path = os.path.join(figures_dir, f"interactive_pca_mollweide_k{k_val}.html")
    with open(html_path, "w") as f:
        f.write(html_template)

    print(f"  Saved interactive visualization to {html_path}")
    print(f"  Generated mollweide projections for {len(projection_images)} cells")


def create_per_dataset_mollweide_visualizations(
    X_pca,
    cluster_labels,
    cell_ids,
    cell_datasets,
    all_results,
    datasets,
    k_val,
    figures_dir,
):
    """
    Create separate interactive visualizations for each dataset.

    Parameters
    ----------
    X_pca : array
        PCA-transformed features
    cluster_labels : array
        Cluster assignments
    cell_ids : list
        Cell IDs
    cell_datasets : list
        Dataset names for each cell
    all_results : list
        Full results containing cell data
    datasets : list
        Unique dataset names
    k_val : int
        Number of clusters
    figures_dir : str
        Directory to save figures
    """
    for dataset in datasets:
        print(f"  Creating interactive visualization for dataset {dataset}...")

        # Filter to cells from this dataset
        dataset_mask = np.array([ds == dataset for ds in cell_datasets])
        dataset_indices = np.where(dataset_mask)[0]

        if len(dataset_indices) == 0:
            continue

        dataset_X_pca = X_pca[dataset_mask]
        dataset_cluster_labels = cluster_labels[dataset_mask]
        dataset_cell_ids = [cell_ids[i] for i in dataset_indices]
        dataset_cell_datasets = [cell_datasets[i] for i in dataset_indices]
        dataset_results = [all_results[i] for i in dataset_indices]

        # Create dataset-specific subdirectory
        dataset_dir = os.path.join(figures_dir, dataset.split("-")[-1])
        os.makedirs(dataset_dir, exist_ok=True)

        # Create interactive visualization for this dataset
        create_interactive_pca_with_projections(
            dataset_X_pca,
            dataset_cluster_labels,
            dataset_cell_ids,
            dataset_cell_datasets,
            dataset_results,
            [dataset],
            k_val,
            dataset_dir,
        )


def create_clustering_visualizations(
    X_scaled,
    X_pca,
    X_umap,
    X_umap_3d,
    cluster_labels,
    k_val,
    cell_ids,
    cell_datasets,
    datasets,
    all_results,
    pca,
    figures_dir,
    feature_names,
    X,
):
    """
    Create all visualizations for a specific k value.

    Parameters
    ----------
    X_scaled : array
        Scaled feature matrix
    X_pca : array
        PCA-transformed features
    X_umap : array
        2D UMAP embedding
    X_umap_3d : array
        3D UMAP embedding
    cluster_labels : array
        Cluster assignments
    k_val : int
        Number of clusters
    cell_ids : list
        Cell IDs
    cell_datasets : list
        Dataset names for each cell
    datasets : list
        Unique dataset names
    all_results : list
        Full results with curves
    pca : PCA object
        Fitted PCA model
    figures_dir : str
        Directory to save figures
    feature_names : list
        Names of features
    X : array
        Raw feature matrix
    """

    # 1. UMAP visualization colored by cluster
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by cluster
    scatter1 = axes[0].scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title(f"Cell clusters (k={k_val})")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    # Color by dataset
    dataset_colors = {ds: i for i, ds in enumerate(datasets)}
    colors = [dataset_colors[ds] for ds in cell_datasets]
    scatter2 = axes[1].scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=colors,
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title("Cells colored by dataset")
    cbar = plt.colorbar(scatter2, ax=axes[1], ticks=range(len(datasets)))
    cbar.ax.set_yticklabels([ds.split("-")[-1] for ds in datasets])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "umap_clusters.png"), dpi=300)
    plt.close()

    # 2. PCA biplot (first 2 components)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"PCA projection of cell features (k={k_val})")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pca_projection.png"), dpi=300)
    plt.close()

    # 3. 3D PCA plot colored by cluster (interactive)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=cluster_labels,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Cluster"),
                    line=dict(width=0.5, color="DarkSlateGray"),
                ),
                text=[
                    f"Cell {cid}<br>Cluster {cl}<br>Dataset {ds}"
                    for cid, cl, ds in zip(cell_ids, cluster_labels, cell_datasets)
                ],
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"3D PCA projection colored by cluster (k={k_val}, interactive)",
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)",
        ),
        width=1000,
        height=800,
    )
    fig.write_html(os.path.join(figures_dir, "pca_3d_by_cluster_interactive.html"))

    # Also save static PNG version
    fig_static = plt.figure(figsize=(12, 10))
    ax = fig_static.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=cluster_labels,
        cmap="tab10",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)")
    ax.set_title(f"3D PCA projection colored by cluster (k={k_val})")
    plt.colorbar(scatter, ax=ax, label="Cluster", shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pca_3d_by_cluster.png"), dpi=300)
    plt.close()

    # 4. 3D UMAP plot colored by cluster (if available)
    if X_umap_3d is not None:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=X_umap_3d[:, 0],
                    y=X_umap_3d[:, 1],
                    z=X_umap_3d[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=cluster_labels,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Cluster"),
                        line=dict(width=0.5, color="DarkSlateGray"),
                    ),
                    text=[
                        f"Cell {cid}<br>Cluster {cl}<br>Dataset {ds}"
                        for cid, cl, ds in zip(cell_ids, cluster_labels, cell_datasets)
                    ],
                    hovertemplate="<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=f"3D UMAP embedding colored by cluster (k={k_val}, interactive)",
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
            width=1000,
            height=800,
        )
        fig.write_html(os.path.join(figures_dir, "umap_3d_by_cluster_interactive.html"))

        # Also save static PNG version
        fig_static = plt.figure(figsize=(12, 10))
        ax = fig_static.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            X_umap_3d[:, 0],
            X_umap_3d[:, 1],
            X_umap_3d[:, 2],
            c=cluster_labels,
            cmap="tab10",
            s=50,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.set_title(f"3D UMAP embedding colored by cluster (k={k_val})")
        plt.colorbar(scatter, ax=ax, label="Cluster", shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "umap_3d_by_cluster.png"), dpi=300)
        plt.close()

    # 5. Per-cluster statistics
    print(f"\n  Cluster statistics for k={k_val}:")
    print("  " + "=" * 78)

    # Save to text file
    stats_file = os.path.join(figures_dir, "cluster_statistics.txt")
    with open(stats_file, "w") as f:
        f.write(f"Cluster statistics for k={k_val}\n")
        f.write("=" * 80 + "\n")

        for cluster_id in range(k_val):
            cluster_mask = cluster_labels == cluster_id
            n_cells = np.sum(cluster_mask)

            output = f"\nCluster {cluster_id} (n={n_cells} cells):\n"
            output += "-" * 40 + "\n"
            f.write(output)
            print("  " + output.replace("\n", "\n  "), end="")

            # Dataset distribution
            cluster_datasets_list = [
                cell_datasets[i] for i in np.where(cluster_mask)[0]
            ]
            for ds in datasets:
                count = cluster_datasets_list.count(ds)
                line = f"  {ds}: {count} cells ({100*count/n_cells:.1f}%)\n"
                f.write(line)
                print("  " + line, end="")

            # Feature means for this cluster
            cluster_features = X[cluster_mask]
            f.write("\n  Feature means:\n")
            print("\n    Feature means:")
            for i, fn in enumerate(feature_names):
                line = f"    {fn}: {np.mean(cluster_features[:, i]):.2f} ± {np.std(cluster_features[:, i]):.2f}\n"
                f.write(line)
                print("  " + line, end="")

    # 6. Representative curves per cluster
    n_cols = min(3, k_val)
    n_rows = (k_val + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if k_val == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if k_val > 1 else [axes]
    else:
        axes = axes.flatten()

    for cluster_id in range(k_val):
        cluster_mask = cluster_labels == cluster_id
        cluster_cells = [all_results[i] for i in np.where(cluster_mask)[0]]

        ax = axes[cluster_id]

        # Plot mean ± std of H(r) for this cluster
        all_H = np.array([c["H"] for c in cluster_cells])
        mean_H = np.mean(all_H, axis=0)
        std_H = np.std(all_H, axis=0)
        r = cluster_cells[0]["r_ripley"]

        ax.plot(r, mean_H, "b-", linewidth=2, label="Mean H(r)")
        ax.fill_between(r, mean_H - std_H, mean_H + std_H, alpha=0.3, color="blue")
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)

        ax.set_xlabel("Geodesic distance (nm)")
        ax.set_ylabel("Ripley's H(r)")
        ax.set_title(f"Cluster {cluster_id} (n={np.sum(cluster_mask)} cells)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(k_val, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cluster_representative_curves.png"), dpi=300)
    plt.close()

    print(f"  Visualizations saved to {figures_dir}/")


def main():
    """Main pipeline for cell-level clustering."""

    datasets = ["jrc_22ak351-leaf-2l", "jrc_22ak351-leaf-3r", "jrc_22ak351-leaf-3m"]

    # Check if we can load existing results
    cache_file = os.path.join(data_dir, "cell_clustering_results.pkl")
    print("=" * 80, flush=True)
    print(
        f"Starting cell clustering analysis..., USE_EXISTING_DATA={USE_EXISTING_DATA}",
        flush=True,
    )
    print(f"Cache file path: {cache_file}", flush=True)
    print(f"Cache file exists: {os.path.exists(cache_file)}", flush=True)
    print("=" * 80, flush=True)

    if USE_EXISTING_DATA and os.path.exists(cache_file):
        print(f"Loading existing results from {cache_file}...")
        with open(cache_file, "rb") as f:
            results_dict = pickle.load(f)

        # Extract variables from loaded results
        cell_ids = results_dict["cell_ids"]
        cell_datasets = results_dict["datasets"]
        feature_names = results_dict["feature_names"]
        X = results_dict["X_raw"]
        X_scaled = results_dict["X_scaled"]
        X_pca = results_dict["X_pca"]
        X_umap = results_dict["X_umap"]
        X_umap_3d = results_dict.get(
            "X_umap_3d", None
        )  # May not exist in old cache files
        cluster_labels = results_dict["cluster_labels"]
        optimal_k = results_dict["optimal_k"]
        silhouette_scores = results_dict["silhouette_scores"]
        scaler = results_dict["scaler"]
        pca = results_dict["pca"]
        reducer = results_dict["umap_reducer"]
        reducer_3d = results_dict.get("umap_reducer_3d", None)
        all_results = results_dict["all_results"]

        print(f"Loaded results for {len(all_results)} cells")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Optimal k: {optimal_k}")
    else:
        # Run full analysis pipeline
        # Bin edges for spatial statistics
        r_bins = np.linspace(0, 20000, 20000 // 500)

        # Process all cells across all datasets
        print("Processing cells across all datasets...")
        all_results = []

        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")

            pkl_pattern = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/geodesic_distances/*_distribution.pkl"
            pkl_files = glob.glob(pkl_pattern)

            print(f"Found {len(pkl_files)} cells")

            # Process cells with Dask for parallelization
            tasks = [
                delayed(process_cell)(pkl_file, r_bins, dataset)
                for pkl_file in pkl_files
            ]

            with ProgressBar():
                results = compute(*tasks, scheduler="threads")

            # Filter valid results
            for result in results:
                if result and "error" not in result:
                    all_results.append(result)
                elif result and "error" in result:
                    print(
                        f"Error processing cell {result['cell_id']} in {result['dataset']}: {result['error']}"
                    )

        print(f"\nSuccessfully processed {len(all_results)} cells total")

        # Build feature matrix
        print("\nBuilding feature matrix...")
        feature_names = list(all_results[0]["features"].keys())
        X = np.array([[r["features"][fn] for fn in feature_names] for r in all_results])

        # Store metadata
        cell_ids = [r["cell_id"] for r in all_results]
        cell_datasets = [r["dataset"] for r in all_results]

        # Handle NaN/Inf in features
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature names: {feature_names}")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA for visualization and noise reduction
        print("\nApplying PCA...")
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        print(
            f"PCA explained variance ratio (first 5 components): {pca.explained_variance_ratio_[:5]}"
        )
        print(
            f"Cumulative explained variance (first 5): {np.cumsum(pca.explained_variance_ratio_[:5])}"
        )

        # UMAP embedding for visualization (2D and 3D)
        print("\nComputing UMAP embeddings...")
        reducer = umap.UMAP(
            n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
        )
        X_umap = reducer.fit_transform(X_scaled)

        reducer_3d = umap.UMAP(
            n_neighbors=15, min_dist=0.1, n_components=3, random_state=42
        )
        X_umap_3d = reducer_3d.fit_transform(X_scaled)

        # Cluster cells
        print("\nClustering cells with KMedoids...")
        K_max = min(20, len(all_results) // 5)  # At least 5 cells per cluster
        silhouette_scores = []

        for k in range(2, K_max + 1):
            km = KMedoids(n_clusters=k, metric="euclidean", random_state=42)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append((k, score))
            print(f"k={k}: silhouette={score:.3f}")

        # Choose optimal k (highest silhouette score)
        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        print(f"\nOptimal number of clusters: {optimal_k}")

        # Fit final clustering model
        km_final = KMedoids(n_clusters=optimal_k, metric="euclidean", random_state=42)
        cluster_labels = km_final.fit_predict(X_scaled)

        # Save results
        print("\nSaving results...")

        # Save feature matrix and metadata
        results_dict = {
            "cell_ids": cell_ids,
            "datasets": cell_datasets,
            "feature_names": feature_names,
            "X_raw": X,
            "X_scaled": X_scaled,
            "X_pca": X_pca,
            "X_umap": X_umap,
            "X_umap_3d": X_umap_3d,
            "cluster_labels": cluster_labels,
            "optimal_k": optimal_k,
            "silhouette_scores": silhouette_scores,
            "scaler": scaler,
            "pca": pca,
            "umap_reducer": reducer,
            "umap_reducer_3d": reducer_3d,
            "all_results": all_results,  # Store full curve data for visualization
        }

        with open(cache_file, "wb") as f:
            pickle.dump(results_dict, f)

        # Create summary DataFrame
        df = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "dataset": cell_datasets,
                "cluster": cluster_labels,
                **{fn: X[:, i] for i, fn in enumerate(feature_names)},
            }
        )
        df.to_csv(os.path.join(data_dir, "cell_clustering_summary.csv"), index=False)

        print(f"Results saved to {data_dir}/")

    # Visualizations
    print("\nCreating visualizations...")

    # 1. Silhouette score vs k
    fig, ax = plt.subplots(figsize=(8, 6))
    ks, scores = zip(*silhouette_scores)
    ax.plot(ks, scores, marker="o")
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
    # Highlight k=2,3,4
    for k_val in [2, 3, 4]:
        if k_val in ks:
            idx = ks.index(k_val)
            ax.plot(k_val, scores[idx], "o", markersize=10, color="orange", zorder=5)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Clustering quality vs number of clusters")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "silhouette_scores.png"), dpi=300)
    plt.close()

    # Generate visualizations for k=2, 3, 4
    print("\nGenerating comparison visualizations for k=2, 3, 4...")
    K_COMPARISON = [2, 3, 4]

    for k_val in K_COMPARISON:
        print(f"\nGenerating visualizations for k={k_val}...")
        k_figures_dir = os.path.join(figures_dir, f"k{k_val}")
        os.makedirs(k_figures_dir, exist_ok=True)

        # Fit clustering model for this k
        km = KMedoids(n_clusters=k_val, metric="euclidean", random_state=42)
        labels_k = km.fit_predict(X_scaled)

        create_clustering_visualizations(
            X_scaled,
            X_pca,
            X_umap,
            X_umap_3d,
            labels_k,
            k_val,
            cell_ids,
            cell_datasets,
            datasets,
            all_results,
            pca,
            k_figures_dir,
            feature_names,
            X,
        )

        # Create interactive PCA visualization with mollweide projections
        create_interactive_pca_with_projections(
            X_pca,
            labels_k,
            cell_ids,
            cell_datasets,
            all_results,
            datasets,
            k_val,
            k_figures_dir,
        )

        # Create per-dataset visualizations
        create_per_dataset_mollweide_visualizations(
            X_pca,
            labels_k,
            cell_ids,
            cell_datasets,
            all_results,
            datasets,
            k_val,
            k_figures_dir,
        )

    print("\n" + "=" * 80)

    # 2. UMAP visualization colored by cluster
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by cluster
    scatter1 = axes[0].scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title(f"Cell clusters (k={optimal_k})")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    # Color by dataset
    dataset_colors = {ds: i for i, ds in enumerate(datasets)}
    colors = [dataset_colors[ds] for ds in cell_datasets]
    scatter2 = axes[1].scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=colors,
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title("Cells colored by dataset")
    cbar = plt.colorbar(scatter2, ax=axes[1], ticks=range(len(datasets)))
    cbar.ax.set_yticklabels([ds.split("-")[-1] for ds in datasets])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "umap_clusters.png"), dpi=300)
    plt.close()

    # 3. PCA biplot (first 2 components)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA projection of cell features")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pca_projection.png"), dpi=300)
    plt.close()

    # 4. 3D PCA plot colored by cluster (interactive)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=cluster_labels,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Cluster"),
                    line=dict(width=0.5, color="DarkSlateGray"),
                ),
                text=[
                    f"Cell {cid}<br>Cluster {cl}<br>Dataset {ds}"
                    for cid, cl, ds in zip(cell_ids, cluster_labels, cell_datasets)
                ],
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="3D PCA projection colored by cluster (interactive)",
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)",
        ),
        width=1000,
        height=800,
    )
    fig.write_html(os.path.join(figures_dir, "pca_3d_by_cluster_interactive.html"))

    # Also save static PNG version
    fig_static = plt.figure(figsize=(12, 10))
    ax = fig_static.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=cluster_labels,
        cmap="tab10",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)")
    ax.set_title("3D PCA projection colored by cluster")
    plt.colorbar(scatter, ax=ax, label="Cluster", shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pca_3d_by_cluster.png"), dpi=300)
    plt.close()

    # 5. 3D PCA plot colored by dataset (interactive)
    dataset_colors_map = {ds: i for i, ds in enumerate(datasets)}
    colors = [dataset_colors_map[ds] for ds in cell_datasets]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=colors,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title="Dataset",
                        tickvals=list(range(len(datasets))),
                        ticktext=[ds.split("-")[-1] for ds in datasets],
                    ),
                    line=dict(width=0.5, color="DarkSlateGray"),
                ),
                text=[
                    f"Cell {cid}<br>Cluster {cl}<br>Dataset {ds}"
                    for cid, cl, ds in zip(cell_ids, cluster_labels, cell_datasets)
                ],
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="3D PCA projection colored by dataset (interactive)",
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)",
        ),
        width=1000,
        height=800,
    )
    fig.write_html(os.path.join(figures_dir, "pca_3d_by_dataset_interactive.html"))

    # Also save static PNG version
    fig_static = plt.figure(figsize=(12, 10))
    ax = fig_static.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=colors,
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)")
    ax.set_title("3D PCA projection colored by dataset")
    cbar = plt.colorbar(
        scatter, ax=ax, ticks=range(len(datasets)), shrink=0.5, aspect=5
    )
    cbar.ax.set_yticklabels([ds.split("-")[-1] for ds in datasets])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pca_3d_by_dataset.png"), dpi=300)
    plt.close()

    # 6. 3D UMAP plot colored by cluster (if available, interactive)
    if X_umap_3d is not None:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=X_umap_3d[:, 0],
                    y=X_umap_3d[:, 1],
                    z=X_umap_3d[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=cluster_labels,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Cluster"),
                        line=dict(width=0.5, color="DarkSlateGray"),
                    ),
                    text=[
                        f"Cell {cid}<br>Cluster {cl}<br>Dataset {ds}"
                        for cid, cl, ds in zip(cell_ids, cluster_labels, cell_datasets)
                    ],
                    hovertemplate="<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="3D UMAP embedding colored by cluster (interactive)",
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
            width=1000,
            height=800,
        )
        fig.write_html(os.path.join(figures_dir, "umap_3d_by_cluster_interactive.html"))

        # Also save static PNG version
        fig_static = plt.figure(figsize=(12, 10))
        ax = fig_static.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            X_umap_3d[:, 0],
            X_umap_3d[:, 1],
            X_umap_3d[:, 2],
            c=cluster_labels,
            cmap="tab10",
            s=50,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.set_title("3D UMAP embedding colored by cluster")
        plt.colorbar(scatter, ax=ax, label="Cluster", shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "umap_3d_by_cluster.png"), dpi=300)
        plt.close()

        # 7. 3D UMAP plot colored by dataset (interactive)
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=X_umap_3d[:, 0],
                    y=X_umap_3d[:, 1],
                    z=X_umap_3d[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(
                            title="Dataset",
                            tickvals=list(range(len(datasets))),
                            ticktext=[ds.split("-")[-1] for ds in datasets],
                        ),
                        line=dict(width=0.5, color="DarkSlateGray"),
                    ),
                    text=[
                        f"Cell {cid}<br>Cluster {cl}<br>Dataset {ds}"
                        for cid, cl, ds in zip(cell_ids, cluster_labels, cell_datasets)
                    ],
                    hovertemplate="<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="3D UMAP embedding colored by dataset (interactive)",
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
            width=1000,
            height=800,
        )
        fig.write_html(os.path.join(figures_dir, "umap_3d_by_dataset_interactive.html"))

        # Also save static PNG version
        fig_static = plt.figure(figsize=(12, 10))
        ax = fig_static.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            X_umap_3d[:, 0],
            X_umap_3d[:, 1],
            X_umap_3d[:, 2],
            c=colors,
            cmap="viridis",
            s=50,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.set_title("3D UMAP embedding colored by dataset")
        cbar = plt.colorbar(
            scatter, ax=ax, ticks=range(len(datasets)), shrink=0.5, aspect=5
        )
        cbar.ax.set_yticklabels([ds.split("-")[-1] for ds in datasets])
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "umap_3d_by_dataset.png"), dpi=300)
        plt.close()

    # 4. Per-cluster statistics
    print("\nCluster statistics:")
    print("=" * 80)

    for cluster_id in range(optimal_k):
        cluster_mask = cluster_labels == cluster_id
        n_cells = np.sum(cluster_mask)

        print(f"\nCluster {cluster_id} (n={n_cells} cells):")
        print("-" * 40)

        # Dataset distribution
        cluster_datasets = [cell_datasets[i] for i in np.where(cluster_mask)[0]]
        for ds in datasets:
            count = cluster_datasets.count(ds)
            print(f"  {ds}: {count} cells ({100*count/n_cells:.1f}%)")

        # Feature means for this cluster
        cluster_features = X[cluster_mask]
        print("\n  Feature means:")
        for i, fn in enumerate(feature_names):
            print(
                f"    {fn}: {np.mean(cluster_features[:, i]):.2f} ± {np.std(cluster_features[:, i]):.2f}"
            )

    # 5. Representative curves per cluster
    n_cols = 3
    n_rows = (optimal_k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if optimal_k > 1 else [axes]

    for cluster_id in range(optimal_k):
        cluster_mask = cluster_labels == cluster_id
        cluster_cells = [all_results[i] for i in np.where(cluster_mask)[0]]

        ax = axes[cluster_id]

        # Plot mean ± std of H(r) for this cluster
        all_H = np.array([c["H"] for c in cluster_cells])
        mean_H = np.mean(all_H, axis=0)
        std_H = np.std(all_H, axis=0)
        r = cluster_cells[0]["r_ripley"]

        ax.plot(r, mean_H, "b-", linewidth=2, label="Mean H(r)")
        ax.fill_between(r, mean_H - std_H, mean_H + std_H, alpha=0.3, color="blue")
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)

        ax.set_xlabel("Geodesic distance (nm)")
        ax.set_ylabel("Ripley's H(r)")
        ax.set_title(f"Cluster {cluster_id} (n={np.sum(cluster_mask)} cells)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(optimal_k, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cluster_representative_curves.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print(f"✓ Cell clustering complete!")
    print(f"✓ Identified {optimal_k} distinct spatial pattern regimes")
    print(f"✓ Results saved to {data_dir}/")
    print(f"✓ Figures saved to {figures_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
