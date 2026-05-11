"""
Investigate what's special about vertices 1594 and 3469 that causes geodesic to fail.
"""

import numpy as np
import pandas as pd
import trimesh
import sys
from pygeodesic import geodesic

sys.path.insert(0, '/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/analysis')
from remesh_and_measure_dask import insert_points_into_mesh_batch, group_plasmodesmata_by_cell

# Load data
cell_meshes_path = "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3r/cell_fixed/meshes/"
plasmodesmata_csv = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/plasmodesmata_lines_assigned_to_2_nearest_cells.csv"
cell_csv = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/cell_fixed.csv"

cell_id = 84

cells_with_plasmodesmata = group_plasmodesmata_by_cell(plasmodesmata_csv, cell_csv)
cell_84_row = cells_with_plasmodesmata[cells_with_plasmodesmata["Cell ID"] == cell_id].iloc[0]
cell_plasmodesmata_coords = np.array(cell_84_row["plasmodesmata_coords"])

mesh_path = f"{cell_meshes_path}/{cell_id}.ply"
cell_mesh = trimesh.load(mesh_path)

print(f"Original mesh: {len(cell_mesh.vertices)} vertices")
print()

updated_vertices, updated_faces, insertion_counts, mapped_indices = insert_points_into_mesh_batch(
    cell_mesh, cell_plasmodesmata_coords
)

print(f"Updated mesh: {len(updated_vertices)} vertices")
print(f"Number of plasmodesmata: {len(cell_plasmodesmata_coords)}")
print(f"Mapped indices range: {min(mapped_indices)} to {max(mapped_indices)}")
print()

# Check if 1594 and 3469 exist
if 1594 >= len(updated_vertices) or 3469 >= len(updated_vertices):
    print("ERROR: Vertices don't exist in this run (non-deterministic behavior)")
    sys.exit(1)

print("="*80)
print("INVESTIGATING VERTEX 1594")
print("="*80)
print()

# Find which plasmodesmata maps to 1594
pd_idx_1594 = [i for i, idx in enumerate(mapped_indices) if idx == 1594]
print(f"Plasmodesmata index: {pd_idx_1594}")

if pd_idx_1594:
    pd_idx = pd_idx_1594[0]
    print(f"Original plasmodesmata coords: {cell_plasmodesmata_coords[pd_idx]}")
    print(f"Final vertex position: {updated_vertices[1594]}")
    print()

    # Check how this point was inserted
    # Was it snapped to existing vertex, on edge, or interior?
    if 1594 < len(cell_mesh.vertices):
        print("Vertex 1594 is an ORIGINAL mesh vertex (snapped to existing)")
    else:
        print("Vertex 1594 was ADDED during insertion")
        # Check insertion count
        if 1594 in insertion_counts:
            print(f"  Insertion count: {insertion_counts[1594]} (multiple PD snapped to this vertex)")
        else:
            print(f"  Insertion count: 1")

print()
print("="*80)
print("INVESTIGATING VERTEX 3469")
print("="*80)
print()

pd_idx_3469 = [i for i, idx in enumerate(mapped_indices) if idx == 3469]
print(f"Plasmodesmata index: {pd_idx_3469}")

if pd_idx_3469:
    pd_idx = pd_idx_3469[0]
    print(f"Original plasmodesmata coords: {cell_plasmodesmata_coords[pd_idx]}")
    print(f"Final vertex position: {updated_vertices[3469]}")
    print()

    if 3469 < len(cell_mesh.vertices):
        print("Vertex 3469 is an ORIGINAL mesh vertex (snapped to existing)")
    else:
        print("Vertex 3469 was ADDED during insertion")
        if 3469 in insertion_counts:
            print(f"  Insertion count: {insertion_counts[3469]} (multiple PD snapped to this vertex)")
        else:
            print(f"  Insertion count: 1")

print()
print("="*80)
print("TESTING GEODESIC DISTANCES FROM/TO THESE VERTICES")
print("="*80)
print()

# Initialize pygeodesic
geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)

# Test different scenarios
print("1. Can we compute distance FROM 1594 TO 3469?")
try:
    dist, _ = geoalg.geodesicDistances([1594], [3469])
    print(f"   Distance: {dist[0]}")
    if not np.isfinite(dist[0]):
        print(f"   FAILED: Distance is {dist[0]}")
except Exception as e:
    print(f"   CRASHED: {e}")

print()
print("2. Can we compute distance FROM 3469 TO 1594?")
try:
    dist, _ = geoalg.geodesicDistances([3469], [1594])
    print(f"   Distance: {dist[0]}")
    if not np.isfinite(dist[0]):
        print(f"   FAILED: Distance is {dist[0]}")
except Exception as e:
    print(f"   CRASHED: {e}")

print()
print("3. Can we compute distance FROM 1594 to nearby vertices?")
# Get vertices in faces containing 1594
faces_with_1594 = np.where(np.any(updated_faces == 1594, axis=1))[0]
nearby_verts = set()
for face_idx in faces_with_1594:
    nearby_verts.update(updated_faces[face_idx])
nearby_verts.discard(1594)
nearby_verts = list(nearby_verts)[:5]  # Just test first 5

print(f"   Testing {len(nearby_verts)} vertices in faces adjacent to 1594:")
for v in nearby_verts:
    try:
        dist, _ = geoalg.geodesicDistances([1594], [v])
        print(f"     1594 -> {v}: {dist[0]:.2f} {'FAIL' if not np.isfinite(dist[0]) else 'OK'}")
    except Exception as e:
        print(f"     1594 -> {v}: CRASH - {e}")

print()
print("4. Can we compute distance FROM 3469 to nearby vertices?")
faces_with_3469 = np.where(np.any(updated_faces == 3469, axis=1))[0]
nearby_verts = set()
for face_idx in faces_with_3469:
    nearby_verts.update(updated_faces[face_idx])
nearby_verts.discard(3469)
nearby_verts = list(nearby_verts)[:5]

print(f"   Testing {len(nearby_verts)} vertices in faces adjacent to 3469:")
for v in nearby_verts:
    try:
        dist, _ = geoalg.geodesicDistances([3469], [v])
        print(f"     3469 -> {v}: {dist[0]:.2f} {'FAIL' if not np.isfinite(dist[0]) else 'OK'}")
    except Exception as e:
        print(f"     3469 -> {v}: CRASH - {e}")

print()
print("5. Test distance from 1594 to a few other mapped indices")
test_indices = [0, 100, 200, 300, 400]
print(f"   Testing distances from vertex 1594 to plasmodesmata indices {test_indices}:")
for pd_i in test_indices:
    if pd_i < len(mapped_indices):
        v = mapped_indices[pd_i]
        try:
            dist, _ = geoalg.geodesicDistances([1594], [v])
            print(f"     1594 (PD {pd_idx_1594[0] if pd_idx_1594 else '?'}) -> {v} (PD {pd_i}): {dist[0]:.2f} {'FAIL' if not np.isfinite(dist[0]) else 'OK'}")
        except Exception as e:
            print(f"     1594 -> {v} (PD {pd_i}): CRASH - {e}")

print()
print("6. Test distance from 3469 to a few other mapped indices")
print(f"   Testing distances from vertex 3469 to plasmodesmata indices {test_indices}:")
for pd_i in test_indices:
    if pd_i < len(mapped_indices):
        v = mapped_indices[pd_i]
        try:
            dist, _ = geoalg.geodesicDistances([3469], [v])
            print(f"     3469 (PD {pd_idx_3469[0] if pd_idx_3469 else '?'}) -> {v} (PD {pd_i}): {dist[0]:.2f} {'FAIL' if not np.isfinite(dist[0]) else 'OK'}")
        except Exception as e:
            print(f"     3469 -> {v} (PD {pd_i}): CRASH - {e}")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print("If 1594 and 3469 can compute distances to nearby vertices but not to each other,")
print("this suggests a path-finding failure in pygeodesic, not a local geometry issue.")
print("If they can't compute to ANY vertices, it's a local geometry/initialization issue.")
