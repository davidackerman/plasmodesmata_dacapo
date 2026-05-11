"""
Analyze why vertices 1594 and 3469 fail geodesic computation.
"""

import numpy as np
import pandas as pd
import trimesh
import sys

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

updated_vertices, updated_faces, insertion_counts, mapped_indices = insert_points_into_mesh_batch(
    cell_mesh, cell_plasmodesmata_coords
)

print(f"Analyzing failed vertices 1594 and 3469")
print(f"="*80)
print()

# Check if these vertices exist
print(f"Total vertices in updated mesh: {len(updated_vertices)}")
print(f"Vertex 1594 exists: {1594 < len(updated_vertices)}")
print(f"Vertex 3469 exists: {3469 < len(updated_vertices)}")
print()

if 1594 >= len(updated_vertices) or 3469 >= len(updated_vertices):
    print("ERROR: One or both vertices don't exist in this run!")
    print("This confirms non-deterministic behavior.")
    sys.exit(1)

# Get vertex positions
v1594 = updated_vertices[1594]
v3469 = updated_vertices[3469]

print(f"Vertex 1594 position: {v1594}")
print(f"Vertex 3469 position: {v3469}")
print(f"Euclidean distance: {np.linalg.norm(v1594 - v3469):.2f} nm")
print()

# Find which plasmodesmata these correspond to
pd_1594 = [i for i, idx in enumerate(mapped_indices) if idx == 1594]
pd_3469 = [i for i, idx in enumerate(mapped_indices) if idx == 3469]

print(f"Vertex 1594 corresponds to plasmodesmata indices: {pd_1594}")
print(f"Vertex 3469 corresponds to plasmodesmata indices: {pd_3469}")
print()

# Check local mesh quality around these vertices
updated_mesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces)

# Find faces containing each vertex
faces_with_1594 = np.where(np.any(updated_faces == 1594, axis=1))[0]
faces_with_3469 = np.where(np.any(updated_faces == 3469, axis=1))[0]

print(f"Faces containing vertex 1594: {len(faces_with_1594)}")
print(f"Faces containing vertex 3469: {len(faces_with_3469)}")
print()

# Check face quality
face_areas_1594 = updated_mesh.area_faces[faces_with_1594]
face_areas_3469 = updated_mesh.area_faces[faces_with_3469]

print(f"Faces around vertex 1594:")
print(f"  Min area: {face_areas_1594.min():.6f}")
print(f"  Max area: {face_areas_1594.max():.6f}")
print(f"  Mean area: {face_areas_1594.mean():.6f}")
print()

print(f"Faces around vertex 3469:")
print(f"  Min area: {face_areas_3469.min():.6f}")
print(f"  Max area: {face_areas_3469.max():.6f}")
print(f"  Mean area: {face_areas_3469.mean():.6f}")
print()

# Try graph-based distance as sanity check
print("Computing graph-based (not geodesic) shortest path...")
try:
    import networkx as nx

    # Build graph from mesh edges
    edges = updated_mesh.edges_unique
    G = nx.Graph()
    for edge in edges:
        v1, v2 = edge
        dist = np.linalg.norm(updated_vertices[v1] - updated_vertices[v2])
        G.add_edge(v1, v2, weight=dist)

    if nx.has_path(G, 1594, 3469):
        path_length = nx.shortest_path_length(G, 1594, 3469, weight='weight')
        print(f"  Graph shortest path distance: {path_length:.2f} nm")
        print(f"  Path exists in graph - mesh IS connected")
    else:
        print(f"  NO PATH in graph - mesh is disconnected!")

        # Find which component each is in
        components = list(nx.connected_components(G))
        for i, comp in enumerate(components):
            if 1594 in comp:
                print(f"  Vertex 1594 in component {i} (size {len(comp)})")
            if 3469 in comp:
                print(f"  Vertex 3469 in component {i} (size {len(comp)})")

except ImportError:
    print("  NetworkX not available, skipping graph analysis")

print()
print("="*80)
print("CONCLUSION:")
print("If graph path exists but geodesic fails, this is a pygeodesic numerical issue,")
print("not a mesh topology problem. Consider:")
print("  1. Using graph-based distances as fallback")
print("  2. Using a different geodesic library (gdist, igl)")
print("  3. Skipping problematic cells")
print("="*80)
