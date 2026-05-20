#%%
#!/usr/bin/env python3
"""
Visualize geodesic distances on a cell mesh surface using Plotly.

Creates an interactive 3D visualization where points (plasmodesmata) on a mesh
are colored by their geodesic distance from a selected source point.
"""

import numpy as np
import trimesh
import plotly.graph_objects as go
import plotly.io as pio
import pickle
from pathlib import Path
import pygeodesic.geodesic as geodesic
from tqdm import tqdm


def load_cell_data(pkl_path):
    """Load pre-computed cell data from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_shrunk_mesh(mesh_path, factor=0.5):
    """
    Load a mesh and shrink it inward along vertex normals.

    This is useful for visualization so the mesh doesn't occlude the points.

    Args:
        mesh_path: Path to the mesh file (e.g., .ply)
        factor: Shrink factor as a multiple of mean edge length (default 0.5)

    Returns:
        trimesh.Trimesh: The shrunk mesh
    """
    mesh = trimesh.load_mesh(mesh_path, process=True, validate=True)
    mesh.remove_unreferenced_vertices()

    normals = mesh.vertex_normals  # outward normals
    # Shrink inward by moving along -normal
    shrunk_verts = mesh.vertices - normals * (mesh.edges_unique_length.mean() * factor)
    shrunk_mesh = trimesh.Trimesh(vertices=shrunk_verts, faces=mesh.faces)
    return shrunk_mesh


def get_mesh_path(dataset, cell_id):
    """
    Get the mesh file path for a given dataset and cell ID.

    Args:
        dataset: Dataset name (e.g., 'jrc_22ak351-leaf-2l', 'jrc_22ak351-leaf-3m', 'jrc_22ak351-leaf-3r')
        cell_id: Cell ID number

    Returns:
        Path to the mesh file
    """
    return f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/{dataset}/cell_fixed/meshes/{cell_id}.ply"


def get_geodesic_pkl_path(dataset, cell_id):
    """
    Get the geodesic distance pickle file path for a given dataset and cell ID.

    Args:
        dataset: Dataset name
        cell_id: Cell ID number

    Returns:
        Path to the geodesic distance pickle file
    """
    return f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/geodesic_distances/{cell_id}_distribution.pkl"


def compute_geodesic_from_source(geoalg, source_idx, num_targets):
    """
    Compute geodesic distances from a single source point to all other points.

    Args:
        geoalg: PyGeodesicAlgorithmExact instance
        source_idx: Index of the source vertex
        num_targets: Total number of vertices

    Returns:
        Array of distances from source to all vertices
    """
    all_targets = list(range(num_targets))
    all_targets.remove(source_idx)

    distances, _ = geoalg.geodesicDistances([source_idx], all_targets)

    # Insert 0 at source position
    full_distances = np.insert(distances, source_idx, 0.0)
    return full_distances


def visualize_mesh_with_geodesic_distance(
    mesh,
    points,
    dist_matrix=None,
    source_idx=0,
    title="Geodesic Distance Visualization",
    colorscale="Viridis",
    point_size=6,
    mesh_opacity=0.3,
    output_html=None
):
    """
    Create an interactive Plotly visualization of a mesh with points colored by
    geodesic distance from a source point.

    Args:
        mesh: trimesh.Trimesh object
        points: Nx3 array of point coordinates on the mesh surface
        dist_matrix: Pre-computed pairwise distance matrix (optional)
        source_idx: Index of the source point (0-indexed into points array)
        title: Plot title
        colorscale: Plotly colorscale name
        point_size: Size of scatter points
        mesh_opacity: Opacity of the mesh (0-1)
        output_html: If provided, save to this HTML file path

    Returns:
        Plotly figure object
    """
    pio.renderers.default = "browser"

    # Get distances from source point
    if dist_matrix is not None:
        distances = dist_matrix[source_idx, :]
    else:
        raise ValueError("dist_matrix must be provided")

    # Handle infinite distances (disconnected points)
    max_finite = np.max(distances[np.isfinite(distances)])
    distances = np.where(np.isfinite(distances), distances, max_finite * 1.1)

    # Create scatter trace for points colored by distance
    scatter_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color=distances,
            colorscale=colorscale,
            colorbar=dict(
                title="Geodesic Distance (nm)",
                titleside="right"
            ),
            opacity=1.0,
        ),
        name="Plasmodesmata",
        text=[f"Point {i}<br>Distance: {d:.1f} nm" for i, d in enumerate(distances)],
        hoverinfo="text"
    )

    # Highlight the source point
    source_trace = go.Scatter3d(
        x=[points[source_idx, 0]],
        y=[points[source_idx, 1]],
        z=[points[source_idx, 2]],
        mode="markers",
        marker=dict(
            size=point_size * 2,
            color="red",
            symbol="diamond",
            line=dict(color="black", width=2)
        ),
        name=f"Source (Point {source_idx})",
        text=f"SOURCE: Point {source_idx}",
        hoverinfo="text"
    )

    # Create mesh trace
    mesh_trace = go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color="lightgray",
        opacity=mesh_opacity,
        name="Cell Mesh",
        hoverinfo="skip"
    )

    # Create figure
    fig = go.Figure(data=[mesh_trace, scatter_trace, source_trace])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title="X (nm)",
            yaxis_title="Y (nm)",
            zaxis_title="Z (nm)",
            aspectmode="data"
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    if output_html:
        fig.write_html(output_html)
        print(f"Saved interactive visualization to {output_html}")

    return fig


def visualize_from_pkl(
    pkl_path,
    source_idx=0,
    output_html=None,
    show=True
):
    """
    Load data from a pickle file and create geodesic distance visualization.

    Args:
        pkl_path: Path to pickle file containing mesh and distance matrix
        source_idx: Index of source point
        output_html: Optional path to save HTML
        show: Whether to display the figure

    Returns:
        Plotly figure
    """
    print(f"Loading data from {pkl_path}...")
    data = load_cell_data(pkl_path)

    # Extract required data
    mesh = data.get("shrunk_mesh") or data.get("mesh")
    dist_matrix = data.get("distance_matrix")
    points = data.get("plasmodesmata_projected") or data.get("plasmodesmata_coords")

    if mesh is None:
        raise ValueError("No mesh found in pickle file")
    if dist_matrix is None:
        raise ValueError("No distance_matrix found in pickle file")
    if points is None:
        raise ValueError("No plasmodesmata coordinates found in pickle file")

    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Points: {len(points)}")
    print(f"  Distance matrix: {dist_matrix.shape}")

    # Create visualization
    fig = visualize_mesh_with_geodesic_distance(
        mesh=mesh,
        points=points,
        dist_matrix=dist_matrix,
        source_idx=source_idx,
        title=f"Geodesic Distance from Point {source_idx}",
        output_html=output_html
    )

    if show:
        fig.show()

    return fig


def create_multi_source_visualization(
    mesh,
    points,
    dist_matrix,
    source_indices,
    title="Multi-Source Geodesic Distance",
    output_html=None
):
    """
    Create visualization showing minimum distance from any of multiple source points.
    Useful for visualizing "distance from nearest plasmodesma" type analyses.

    Args:
        mesh: trimesh.Trimesh object
        points: Nx3 array of point coordinates
        dist_matrix: Pairwise distance matrix
        source_indices: List of source point indices
        title: Plot title
        output_html: Optional HTML output path

    Returns:
        Plotly figure
    """
    pio.renderers.default = "browser"

    # Compute minimum distance from any source
    source_distances = dist_matrix[source_indices, :]
    min_distances = np.min(source_distances, axis=0)

    # Handle infinite distances
    max_finite = np.max(min_distances[np.isfinite(min_distances)])
    min_distances = np.where(np.isfinite(min_distances), min_distances, max_finite * 1.1)

    # Create scatter trace
    scatter_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=6,
            color=min_distances,
            colorscale="Viridis",
            colorbar=dict(title="Min Distance to Source (nm)"),
            opacity=1.0,
        ),
        name="Points",
        text=[f"Point {i}<br>Min Distance: {d:.1f} nm" for i, d in enumerate(min_distances)],
        hoverinfo="text"
    )

    # Highlight source points
    source_trace = go.Scatter3d(
        x=points[source_indices, 0],
        y=points[source_indices, 1],
        z=points[source_indices, 2],
        mode="markers",
        marker=dict(
            size=10,
            color="red",
            symbol="diamond",
        ),
        name=f"Sources ({len(source_indices)} points)",
    )

    # Mesh trace
    mesh_trace = go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color="lightgray",
        opacity=0.3,
        name="Cell Mesh",
        hoverinfo="skip"
    )

    fig = go.Figure(data=[mesh_trace, scatter_trace, source_trace])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (nm)",
            yaxis_title="Y (nm)",
            zaxis_title="Z (nm)",
            aspectmode="data"
        )
    )

    if output_html:
        fig.write_html(output_html)
        print(f"Saved to {output_html}")

    return fig


def main():
    """Example usage with sample data."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize geodesic distances on mesh")
    parser.add_argument("pkl_path", help="Path to pickle file with mesh and distance data")
    parser.add_argument("--source", type=int, default=0, help="Source point index")
    parser.add_argument("--output", help="Output HTML file path")
    parser.add_argument("--no-show", action="store_true", help="Don't display figure")

    args = parser.parse_args()

    visualize_from_pkl(
        pkl_path=args.pkl_path,
        source_idx=args.source,
        output_html=args.output,
        show=not args.no_show
    )


if __name__ == "__main__":
    main()

# %% Interactive mode example
# Set these variables for interactive use in a notebook/IDE

# Dataset and cell configuration
DATASET = "jrc_22ak351-leaf-3m"  # Options: jrc_22ak351-leaf-2l, jrc_22ak351-leaf-3m, jrc_22ak351-leaf-3r
CELL_ID = 100

# Source point index to measure distances from
SOURCE_IDX = 0

# Mesh shrink factor (how much to shrink the mesh inward for visualization)
SHRINK_FACTOR = 0.5

# %% Load data and mesh (run this cell first)
# Load geodesic distance data
pkl_path = get_geodesic_pkl_path(DATASET, CELL_ID)
print(f"Loading geodesic data from: {pkl_path}")
data = load_cell_data(pkl_path)

# Extract distance matrix and plasmodesmata coordinates
dist_matrix = data["distance_matrix"]
plasmodesmata_indices = data["plasmodesmata_indices"]
updated_vertices = data["updated_vertices"]
points = updated_vertices[plasmodesmata_indices, :]

# Load and shrink the mesh
mesh_path = get_mesh_path(DATASET, CELL_ID)
print(f"Loading mesh from: {mesh_path}")
shrunk_mesh = get_shrunk_mesh(mesh_path, factor=SHRINK_FACTOR)

print(f"\nLoaded data:")
print(f"  Dataset: {DATASET}")
print(f"  Cell ID: {CELL_ID}")
print(f"  Mesh: {len(shrunk_mesh.vertices)} vertices, {len(shrunk_mesh.faces)} faces")
print(f"  Points (plasmodesmata): {len(points)}")
print(f"  Distance matrix: {dist_matrix.shape}")

# %% Visualize geodesic distance from a single source point
fig = visualize_mesh_with_geodesic_distance(
    mesh=shrunk_mesh,
    points=points,
    dist_matrix=dist_matrix,
    source_idx=SOURCE_IDX,
    title=f"{DATASET} Cell {CELL_ID}: Geodesic Distance from Point {SOURCE_IDX}"
)
fig.show()

# %% Explore different source points interactively
# Change source_idx to explore distances from different points
source_idx = 10

fig = visualize_mesh_with_geodesic_distance(
    mesh=shrunk_mesh,
    points=points,
    dist_matrix=dist_matrix,
    source_idx=source_idx,
    title=f"{DATASET} Cell {CELL_ID}: Geodesic Distance from Point {source_idx}"
)
fig.show()

# %% Multi-source visualization example
# Select multiple source points to show minimum distance from any of them
source_indices = [0, 5, 10, 15]

fig = create_multi_source_visualization(
    mesh=shrunk_mesh,
    points=points,
    dist_matrix=dist_matrix,
    source_indices=source_indices,
    title=f"{DATASET} Cell {CELL_ID}: Min Distance from {len(source_indices)} Sources"
)
fig.show()

# %% Quick function to load and visualize a different cell
def visualize_cell(dataset, cell_id, source_idx=0, shrink_factor=0.5):
    """
    Quick helper to load and visualize geodesic distances for any cell.

    Args:
        dataset: Dataset name (e.g., 'jrc_22ak351-leaf-3m')
        cell_id: Cell ID number
        source_idx: Source point index
        shrink_factor: Mesh shrink factor

    Returns:
        Plotly figure
    """
    # Load data
    pkl_path = get_geodesic_pkl_path(dataset, cell_id)
    data = load_cell_data(pkl_path)

    dist_matrix = data["distance_matrix"]
    plasmodesmata_indices = data["plasmodesmata_indices"]
    updated_vertices = data["updated_vertices"]
    points = updated_vertices[plasmodesmata_indices, :]

    # Load and shrink mesh
    mesh_path = get_mesh_path(dataset, cell_id)
    shrunk_mesh = get_shrunk_mesh(mesh_path, factor=shrink_factor)

    print(f"Cell {cell_id}: {len(points)} plasmodesmata")

    # Visualize
    fig = visualize_mesh_with_geodesic_distance(
        mesh=shrunk_mesh,
        points=points,
        dist_matrix=dist_matrix,
        source_idx=source_idx,
        title=f"{dataset} Cell {cell_id}: Geodesic Distance from Point {source_idx}"
    )
    return fig

# %% Example: visualize a different cell
# fig = visualize_cell("jrc_22ak351-leaf-2l", cell_id=50, source_idx=0)
# fig.show()

# %%
