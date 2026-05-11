"""
Verify if the reverse direction distance is reasonable/correct.
Compare geodesic vs graph vs Euclidean distances.
"""

import numpy as np
import pandas as pd
import trimesh
import sys
from pygeodesic import geodesic
import networkx as nx

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

print(f"Verifying distance between vertices 1594 and 3469")
print(f"="*80)
print()

v1594 = updated_vertices[1594]
v3469 = updated_vertices[3469]

# 1. Euclidean (straight line) distance
euclidean_dist = np.linalg.norm(v1594 - v3469)
print(f"1. Euclidean (straight-line) distance: {euclidean_dist:.2f} nm")
print()

# 2. Graph distance (NetworkX)
print(f"2. Graph distance (shortest path through edges):")
updated_mesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces)
edges = updated_mesh.edges_unique

G = nx.Graph()
for edge in edges:
    v1, v2 = edge
    dist = np.linalg.norm(updated_vertices[v1] - updated_vertices[v2])
    G.add_edge(int(v1), int(v2), weight=dist)

try:
    graph_dist = nx.shortest_path_length(G, 1594, 3469, weight='weight')
    print(f"   Graph distance: {graph_dist:.2f} nm")

    # Get the path
    path = nx.shortest_path(G, 1594, 3469, weight='weight')
    print(f"   Path length: {len(path)} vertices")
    print(f"   Path (first 10): {path[:10]}")
except nx.NetworkXNoPath:
    print(f"   NO PATH - vertices are disconnected!")
    graph_dist = None

print()

# 3. Geodesic distances
print(f"3. Pygeodesic distances:")
geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)

# Forward
try:
    dist_forward, _ = geoalg.geodesicDistances([1594], [3469])
    print(f"   1594 -> 3469: {dist_forward[0]:.2f} nm")
except Exception as e:
    print(f"   1594 -> 3469: CRASHED - {e}")
    dist_forward = [np.inf]

# Reverse
try:
    dist_reverse, _ = geoalg.geodesicDistances([3469], [1594])
    print(f"   3469 -> 1594: {dist_reverse[0]:.2f} nm")
except Exception as e:
    print(f"   3469 -> 1594: CRASHED - {e}")
    dist_reverse = [np.inf]

print()

# 4. Try a few intermediate test points
print(f"4. Test geodesic to intermediate vertices along the graph path:")
if graph_dist and len(path) > 2:
    # Pick a few intermediate points
    test_points = [path[len(path)//4], path[len(path)//2], path[3*len(path)//4]]
    print(f"   Testing through intermediate vertices: {test_points}")
    print()

    for mid_v in test_points:
        # 1594 -> mid
        try:
            d1, _ = geoalg.geodesicDistances([1594], [mid_v])
            print(f"   1594 -> {mid_v}: {d1[0]:.2f} nm")
        except:
            print(f"   1594 -> {mid_v}: CRASH")

        # mid -> 3469
        try:
            d2, _ = geoalg.geodesicDistances([mid_v], [3469])
            print(f"   {mid_v} -> 3469: {d2[0]:.2f} nm")
        except:
            print(f"   {mid_v} -> 3469: CRASH")

        # Total
        if np.isfinite(d1[0]) and np.isfinite(d2[0]):
            print(f"   Total via {mid_v}: {d1[0] + d2[0]:.2f} nm")
        print()

print(f"="*80)
print("COMPARISON:")
print(f"="*80)
print(f"Euclidean (minimum possible): {euclidean_dist:.2f} nm")
if graph_dist:
    print(f"Graph (upper bound):          {graph_dist:.2f} nm")
print(f"Geodesic forward (1594->3469): {dist_forward[0]:.2f} nm")
print(f"Geodesic reverse (3469->1594): {dist_reverse[0]:.2f} nm")
print()

if np.isfinite(dist_reverse[0]):
    # Check if reverse is reasonable
    print("ANALYSIS:")

    # Geodesic should be >= Euclidean
    if dist_reverse[0] >= euclidean_dist:
        print(f"✓ Reverse distance ({dist_reverse[0]:.2f}) >= Euclidean ({euclidean_dist:.2f})")
    else:
        print(f"✗ PROBLEM: Reverse distance ({dist_reverse[0]:.2f}) < Euclidean ({euclidean_dist:.2f})")
        print(f"  This is impossible! Geodesic must be at least as long as straight line.")

    # Geodesic should be <= Graph
    if graph_dist and dist_reverse[0] <= graph_dist:
        print(f"✓ Reverse distance ({dist_reverse[0]:.2f}) <= Graph ({graph_dist:.2f})")
        pct_diff = (graph_dist - dist_reverse[0]) / graph_dist * 100
        print(f"  Geodesic is {pct_diff:.1f}% shorter than graph path (expected)")
    elif graph_dist:
        print(f"✗ PROBLEM: Reverse distance ({dist_reverse[0]:.2f}) > Graph ({graph_dist:.2f})")
        print(f"  This is suspicious! Geodesic should be at most as long as edge path.")

    # Overall verdict
    print()
    if graph_dist and euclidean_dist <= dist_reverse[0] <= graph_dist:
        print("VERDICT: Reverse distance appears REASONABLE")
        print(f"  It falls within the expected bounds: {euclidean_dist:.2f} <= {dist_reverse[0]:.2f} <= {graph_dist:.2f}")
    else:
        print("VERDICT: Reverse distance is SUSPICIOUS - may also be wrong")

print()
print("="*80)
print("Why might pygeodesic fail asymmetrically?")
print("="*80)
print("Possible causes:")
print("1. Internal data structure corruption during path-finding")
print("2. Numerical precision issues that depend on traversal direction")
print("3. Bug in the Fast Marching Method implementation")
print("4. Edge case in the priority queue or wavefront propagation")
print()
print("The fact that reverse works suggests it's not a mesh topology issue,")
print("but rather an algorithmic bug in pygeodesic's path-finding.")
