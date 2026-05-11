"""
Debug script to reproduce the exact geodesic error from cell 84.
Tests geodesic computation multiple times to check for non-determinism.
"""

import numpy as np
import trimesh
import pandas as pd
import os
import sys

# Import functions from the main script
sys.path.insert(0, '/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/analysis')
from remesh_and_measure_dask import insert_points_into_mesh_batch, compute_pairwise_geodesic_for_inputs

# Paths from config
cell_meshes_path = "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3r/cell_fixed/meshes/"
plasmodesmata_csv = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/plasmodesmata_lines_assigned_to_2_nearest_cells.csv"
cell_csv = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/cell_fixed.csv"

cell_id = 84

print(f"\n{'='*80}")
print(f"DEBUGGING CELL {cell_id} - GEODESIC COMPUTATION")
print(f"{'='*80}\n")

# Load the mesh
mesh_path = f"{cell_meshes_path}/{cell_id}.ply"
mesh = trimesh.load(mesh_path)

# Load and process plasmodesmata data
plasmodesmata_df = pd.read_csv(plasmodesmata_csv)
import ast
for list_column in ["Cell ID", "Cell Distance (nm)"]:
    plasmodesmata_df[list_column] = plasmodesmata_df[list_column].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

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

cell_plasmodesmata = exploded_df[exploded_df["Cell ID"] == cell_id]
plasmodesmata_coords = cell_plasmodesmata[
    ["Plasmodesmata COM Z (nm)", "Plasmodesmata COM Y (nm)", "Plasmodesmata COM X (nm)"]
].values

print(f"Plasmodesmata: {len(plasmodesmata_coords)}")

# Perform the insertion
print("Performing point insertion...")
updated_vertices, updated_faces, insertion_counts, mapped_indices = insert_points_into_mesh_batch(
    mesh, plasmodesmata_coords
)

print(f"Updated mesh: {len(updated_vertices)} vertices, {len(updated_faces)} faces")
print(f"Mapped indices range: {min(mapped_indices)} to {max(mapped_indices)}")

# The error mentioned vertices 1594 and 3469
print(f"\nChecking error vertices from original traceback:")
print(f"  Vertex 1594: exists={1594 < len(updated_vertices)}")
print(f"  Vertex 3469: exists={3469 < len(updated_vertices)}")

if 1594 < len(updated_vertices) and 3469 < len(updated_vertices):
    print(f"  Both vertices exist in this run!")
else:
    print(f"  ERROR: One or both vertices don't exist - different behavior than failed run")
    sys.exit(0)

# Find which plasmodesmata these correspond to
pd_1594 = [i for i, idx in enumerate(mapped_indices) if idx == 1594]
pd_3469 = [i for i, idx in enumerate(mapped_indices) if idx == 3469]

print(f"\n  Vertex 1594 corresponds to plasmodesmata: {pd_1594}")
print(f"  Vertex 3469 corresponds to plasmodesmata: {pd_3469}")

# Try computing geodesics
print(f"\n{'='*80}")
print("TESTING GEODESIC COMPUTATION")
print(f"{'='*80}\n")

# Run multiple times to check for non-determinism
num_trials = 3
for trial in range(num_trials):
    print(f"\nTrial {trial + 1}/{num_trials}:")
    try:
        dist_matrix = compute_pairwise_geodesic_for_inputs(
            updated_vertices, updated_faces, mapped_indices
        )
        print(f"  SUCCESS: Distance matrix shape {dist_matrix.shape}")
        print(f"  Max distance: {dist_matrix.max():.2f}")
        print(f"  Mean distance: {dist_matrix.mean():.2f}")

        # Check for any infinite distances
        inf_mask = ~np.isfinite(dist_matrix)
        num_inf = np.sum(inf_mask)
        if num_inf > 0:
            print(f"  WARNING: Found {num_inf} infinite distances!")
            # Find which pairs
            inf_pairs = np.where(inf_mask)
            print(f"  First few infinite pairs:")
            for i in range(min(5, len(inf_pairs[0]))):
                src_idx = inf_pairs[0][i]
                tgt_idx = inf_pairs[1][i]
                src_v = mapped_indices[src_idx]
                tgt_v = mapped_indices[tgt_idx]
                print(f"    PD {src_idx} (vertex {src_v}) -> PD {tgt_idx} (vertex {tgt_v}): inf")

    except ValueError as e:
        print(f"  FAILURE: {e}")

        # Try to diagnose further
        print(f"\n  Checking mesh connectivity...")
        updated_mesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces)
        split_meshes = updated_mesh.split(only_watertight=False)
        print(f"  Number of components: {len(split_meshes)}")

        if len(split_meshes) > 1:
            print(f"  DISCONNECTED MESH DETECTED!")
            for i, sm in enumerate(split_meshes):
                print(f"    Component {i}: {len(sm.vertices)} vertices, {len(sm.faces)} faces")

        break

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")
