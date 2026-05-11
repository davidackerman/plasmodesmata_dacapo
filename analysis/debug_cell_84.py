"""
Debug script to investigate the geodesic distance issue with cell 84.
Identifies which plasmodesmata insertion caused disconnected mesh components.
"""

import numpy as np
import trimesh
import pandas as pd
import os
import sys

# Import the insertion function from the main script
sys.path.insert(0, '/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/analysis')
from remesh_and_measure_dask import insert_points_into_mesh_batch

# Paths from config
cell_meshes_path = "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3r/cell_fixed/meshes/"
plasmodesmata_csv = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/plasmodesmata_lines_assigned_to_2_nearest_cells.csv"
cell_csv = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/cell_fixed.csv"

cell_id = 84

print(f"\n{'='*80}")
print(f"DEBUGGING CELL {cell_id}")
print(f"{'='*80}\n")

# Load the mesh
mesh_path = f"{cell_meshes_path}/{cell_id}.ply"
print(f"Loading mesh from: {mesh_path}")
mesh = trimesh.load(mesh_path)

print(f"\nOriginal mesh properties:")
print(f"  Vertices: {len(mesh.vertices)}")
print(f"  Faces: {len(mesh.faces)}")
print(f"  Is watertight: {mesh.is_watertight}")
print(f"  Is winding consistent: {mesh.is_winding_consistent}")

# Check connectivity
split_meshes = mesh.split(only_watertight=False)
print(f"  Number of connected components: {len(split_meshes)}")
if len(split_meshes) > 1:
    print(f"  WARNING: Original mesh has {len(split_meshes)} disconnected components!")
    for i, sm in enumerate(split_meshes):
        print(f"    Component {i}: {len(sm.vertices)} vertices, {len(sm.faces)} faces")

# Load plasmodesmata data
print(f"\nLoading plasmodesmata from: {plasmodesmata_csv}")
plasmodesmata_df = pd.read_csv(plasmodesmata_csv)

# Parse list columns
import ast
for list_column in ["Cell ID", "Cell Distance (nm)"]:
    plasmodesmata_df[list_column] = plasmodesmata_df[list_column].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

# Explode "Cell ID" so each cell appears in its own row
exploded_df = plasmodesmata_df.explode("Cell ID")
exploded_df = exploded_df.rename(
    columns={
        "COM X (nm)": "Plasmodesmata COM X (nm)",
        "COM Y (nm)": "Plasmodesmata COM Y (nm)",
        "COM Z (nm)": "Plasmodesmata COM Z (nm)",
    }
)
exploded_df = exploded_df[
    [
        "Cell ID",
        "Plasmodesmata COM X (nm)",
        "Plasmodesmata COM Y (nm)",
        "Plasmodesmata COM Z (nm)",
    ]
]

# Load cell data
print(f"Loading cell data from: {cell_csv}")
cell_df = pd.read_csv(cell_csv)
cell_df = cell_df.rename(
    columns={
        "Object ID": "Cell ID",
        "COM X (nm)": "Cell COM X (nm)",
        "COM Y (nm)": "Cell COM Y (nm)",
        "COM Z (nm)": "Cell COM Z (nm)",
    }
)

# Get plasmodesmata for this cell
cell_plasmodesmata = exploded_df[exploded_df["Cell ID"] == cell_id]

print(f"\nPlasmodesmata assigned to cell {cell_id}: {len(cell_plasmodesmata)}")

# Extract coordinates (Z, Y, X order)
plasmodesmata_coords = cell_plasmodesmata[
    ["Plasmodesmata COM Z (nm)", "Plasmodesmata COM Y (nm)", "Plasmodesmata COM X (nm)"]
].values
print(f"Plasmodesmata coordinates shape: {plasmodesmata_coords.shape}")

# Check distances from mesh
print(f"\nChecking distances from plasmodesmata to mesh surface:")
closest, dists, fids = mesh.nearest.on_surface(plasmodesmata_coords)

print(f"  Min distance: {dists.min():.4f}")
print(f"  Max distance: {dists.max():.4f}")
print(f"  Mean distance: {dists.mean():.4f}")
print(f"  Median distance: {np.median(dists):.4f}")

# Find outliers
outlier_threshold = np.percentile(dists, 95)
outliers = np.where(dists > outlier_threshold)[0]
print(f"\n  Distances > 95th percentile ({outlier_threshold:.4f}):")
if len(outliers) > 0:
    for idx in outliers[:10]:  # Show first 10
        print(f"    PD {idx}: distance={dists[idx]:.4f}, face={fids[idx]}, coords={plasmodesmata_coords[idx]}")
else:
    print("    None")

# Now perform the insertion
print(f"\n{'='*80}")
print("PERFORMING POINT INSERTION")
print(f"{'='*80}\n")

updated_vertices, updated_faces, insertion_counts, mapped_indices = insert_points_into_mesh_batch(
    mesh, plasmodesmata_coords
)

print(f"Updated mesh properties:")
print(f"  Vertices: {len(updated_vertices)} (added {len(updated_vertices) - len(mesh.vertices)})")
print(f"  Faces: {len(updated_faces)} (added {len(updated_faces) - len(mesh.faces)})")

# Create a trimesh from updated geometry
updated_mesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces)

print(f"  Is watertight: {updated_mesh.is_watertight}")
print(f"  Is winding consistent: {updated_mesh.is_winding_consistent}")

# Check connectivity of updated mesh
split_updated = updated_mesh.split(only_watertight=False)
print(f"  Number of connected components: {len(split_updated)}")

if len(split_updated) > 1:
    print(f"\n  PROBLEM FOUND: Updated mesh has {len(split_updated)} disconnected components!")
    for i, sm in enumerate(split_updated):
        print(f"    Component {i}: {len(sm.vertices)} vertices, {len(sm.faces)} faces")

    # Find which plasmodesmata vertices are in which component
    print(f"\n  Analyzing which plasmodesmata are in which component:")

    # Get vertex to component mapping
    vertex_to_component = {}
    for comp_idx, sm in enumerate(split_updated):
        # Get the vertices in this component by checking positions
        for v_idx, v_pos in enumerate(updated_vertices):
            v_tuple = tuple(v_pos)
            for sm_v in sm.vertices:
                if np.allclose(v_pos, sm_v, atol=1e-9):
                    vertex_to_component[v_idx] = comp_idx
                    break

    # Check which component each plasmodesmata is in
    pd_components = {}
    for pd_idx, mesh_v_idx in enumerate(mapped_indices):
        comp = vertex_to_component.get(mesh_v_idx, -1)
        if comp not in pd_components:
            pd_components[comp] = []
        pd_components[comp].append(pd_idx)

    print(f"\n  Plasmodesmata distribution across components:")
    for comp, pd_list in sorted(pd_components.items()):
        print(f"    Component {comp}: {len(pd_list)} plasmodesmata")
        if len(pd_list) <= 5:
            print(f"      PD indices: {pd_list}")
            for pd_idx in pd_list:
                print(f"        PD {pd_idx}: vertex {mapped_indices[pd_idx]}, distance={dists[pd_idx]:.4f}")

    # The problematic vertices from the error message
    print(f"\n  Checking error vertices:")
    print(f"    Vertex 1594 in component: {vertex_to_component.get(1594, 'NOT FOUND')}")
    print(f"    Vertex 3469 in component: {vertex_to_component.get(3469, 'NOT FOUND')}")

    # Find which plasmodesmata correspond to these vertices
    for pd_idx, mesh_v_idx in enumerate(mapped_indices):
        if mesh_v_idx == 1594:
            print(f"    Vertex 1594 is plasmodesmata {pd_idx}")
            print(f"      Distance from mesh: {dists[pd_idx]:.4f}")
            print(f"      Coords: {plasmodesmata_coords[pd_idx]}")
        if mesh_v_idx == 3469:
            print(f"    Vertex 3469 is plasmodesmata {pd_idx}")
            print(f"      Distance from mesh: {dists[pd_idx]:.4f}")
            print(f"      Coords: {plasmodesmata_coords[pd_idx]}")

else:
    print("\n  SUCCESS: Updated mesh is fully connected!")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")
