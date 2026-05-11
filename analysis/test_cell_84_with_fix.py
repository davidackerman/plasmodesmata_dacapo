"""
Test cell 84 with the new geodesic robustness fixes.
"""

import numpy as np
import pandas as pd
import trimesh
import sys
import os

# Import the fixed functions
sys.path.insert(0, '/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/analysis')
from remesh_and_measure_dask import (
    insert_points_into_mesh_batch,
    compute_pairwise_geodesic_for_inputs,
    group_plasmodesmata_by_cell
)

# Paths from config
cell_meshes_path = "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3r/cell_fixed/meshes/"
plasmodesmata_csv = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/plasmodesmata_lines_assigned_to_2_nearest_cells.csv"
cell_csv = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/cell_fixed.csv"
output_path = "/tmp/test_cell_84_output"

cell_id = 84

print(f"\n{'='*80}")
print(f"TESTING CELL {cell_id} WITH NEW GEODESIC FIX")
print(f"{'='*80}\n")

# Load data using the same method as the main script
print("Loading plasmodesmata and cell data...")
cells_with_plasmodesmata = group_plasmodesmata_by_cell(plasmodesmata_csv, cell_csv)

# Get cell 84 data
cell_84_row = cells_with_plasmodesmata[cells_with_plasmodesmata["Cell ID"] == cell_id].iloc[0]
cell_plasmodesmata_coords = np.array(cell_84_row["plasmodesmata_coords"])

print(f"Cell {cell_id}:")
print(f"  Plasmodesmata count: {len(cell_plasmodesmata_coords)}")
print(f"  Coordinates shape: {cell_plasmodesmata_coords.shape}")
print(f"  Coordinates dtype: {cell_plasmodesmata_coords.dtype}")
print()

# Load mesh
mesh_path = f"{cell_meshes_path}/{cell_id}.ply"
print(f"Loading mesh from: {mesh_path}")
cell_mesh = trimesh.load(mesh_path)

print(f"Original mesh:")
print(f"  Vertices: {len(cell_mesh.vertices)}")
print(f"  Faces: {len(cell_mesh.faces)}")
print(f"  Is watertight: {cell_mesh.is_watertight}")
print(f"  Is connected: {cell_mesh.body_count == 1}")
print()

# Insert points into mesh
print("Inserting plasmodesmata into mesh...")
updated_vertices, updated_faces, insertion_counts, mapped_indices = (
    insert_points_into_mesh_batch(cell_mesh, cell_plasmodesmata_coords)
)

print(f"Updated mesh:")
print(f"  Vertices: {len(updated_vertices)} (added {len(updated_vertices) - len(cell_mesh.vertices)})")
print(f"  Faces: {len(updated_faces)} (added {len(updated_faces) - len(cell_mesh.faces)})")
print()

# Check if updated mesh is connected
updated_mesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces)
print(f"Updated mesh properties:")
print(f"  Is watertight: {updated_mesh.is_watertight}")
print(f"  Number of components: {updated_mesh.body_count}")
print()

# Compute geodesic distances with the new robust function
print("Computing geodesic distances with new robust implementation...")
print()

try:
    dist_matrix = compute_pairwise_geodesic_for_inputs(
        updated_vertices, updated_faces, mapped_indices, cell_id=str(cell_id)
    )

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}\n")

    print(f"Distance matrix computed successfully:")
    print(f"  Shape: {dist_matrix.shape}")
    print(f"  Min distance: {dist_matrix[dist_matrix > 0].min():.2f} nm")
    print(f"  Max distance: {dist_matrix.max():.2f} nm")
    print(f"  Mean distance: {dist_matrix[dist_matrix > 0].mean():.2f} nm")

    # Check for any issues
    if np.any(~np.isfinite(dist_matrix)):
        print(f"\n  WARNING: Found non-finite values in distance matrix!")
    else:
        print(f"\n  All distances are finite ✓")

    if not np.allclose(dist_matrix, dist_matrix.T):
        print(f"  WARNING: Distance matrix is not symmetric!")
    else:
        print(f"  Distance matrix is symmetric ✓")

    if not np.allclose(np.diag(dist_matrix), 0):
        print(f"  WARNING: Diagonal is not zero!")
    else:
        print(f"  Diagonal is zero ✓")

except ValueError as e:
    print(f"\n{'='*80}")
    print("FAILURE!")
    print(f"{'='*80}\n")
    print(f"Error: {e}")
    print()
    print("The enhanced error message above should provide details about the failure.")

except Exception as e:
    print(f"\n{'='*80}")
    print("UNEXPECTED ERROR!")
    print(f"{'='*80}\n")
    print(f"Error type: {type(e).__name__}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}\n")
