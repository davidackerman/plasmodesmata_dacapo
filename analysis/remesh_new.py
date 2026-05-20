# %%
# Batch surface point insertion (by face & edge) using trimesh for projection
# and Triangle (if available) for fast per-face triangulation.
# Key improvements over the previous incremental version:
#   - One global projection pass (trimesh.nearest.on_surface) for all points
#   - Dedup upfront (against existing vertices and among new points)
#   - Batch edge processing: split shared edges once using ordered chains
#   - Per-face re-triangulation in 2D with boundary constraints (Triangle)
#   - Single mesh rebuild at the end; optional validity checks
#   - Still returns mapped_indices (one per input) and insertion_counts

import numpy as np
import trimesh
from collections import defaultdict
from pygeodesic import geodesic
from tqdm import tqdm

# Optional fast 2D constrained triangulation
try:
    import triangle as tr  # pip install triangle

    _HAS_TRIANGLE = True
except Exception:  # pragma: no cover
    from scipy.spatial import Delaunay  # fallback

    _HAS_TRIANGLE = False

# ---------------------------- helpers ---------------------------------


def build_trimesh(vertices, faces):
    return trimesh.Trimesh(
        vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=int), process=False
    )


def face_frames(V, F):
    """Precompute local frames (origin A, ex, ey, normal) per face."""
    V = np.asarray(V)
    A = V[F[:, 0]]
    B = V[F[:, 1]]
    C = V[F[:, 2]]
    e0 = B - A
    n = np.cross(e0, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    ex = e0 / np.maximum(np.linalg.norm(e0, axis=1, keepdims=True), 1e-15)
    ey = np.cross(n, ex)
    return A, ex, ey, n


def to_local2(A, ex, ey, P):
    """Project 3D points P to local 2D coords w.r.t. (A, ex, ey)."""
    d = P - A
    x = np.dot(d, ex)
    y = np.dot(d, ey)
    return np.stack([x, y], axis=-1)


def barycentric_in_face(P, A, B, C):
    v0, v1, v2 = B - A, C - A, P - A
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def remove_degenerate_and_duplicate_faces(vertices, faces, area_tol=1e-14):
    unique = {}
    cleaned = []
    V = np.asarray(vertices)
    for f in faces:
        i, j, k = f
        if i == j or j == k or k == i:
            continue
        a, b, c = V[i], V[j], V[k]
        area2 = np.linalg.norm(np.cross(b - a, c - a))
        if not np.isfinite(area2) or area2 < area_tol:
            continue
        key = tuple(sorted((i, j, k)))
        if key in unique:
            continue
        unique[key] = f
        cleaned.append(f)
    return cleaned


# ---------------------------- main API --------------------------------


def insert_points_into_mesh_batch(
    mesh: trimesh.Trimesh,
    new_points,
    *,
    tol=1e-8,
    round_dp=9,
    use_triangle=True,
    validate=False,
):
    """
    Batch insert "new_points" onto a triangular surface mesh without incremental splits.

    Steps:
      1) Project all points to the surface once.
      2) Snap to existing vertices / shared edges; dedup among new points.
      3) Build per-edge chains and per-face interior point lists.
      4) Re-triangulate each affected face in 2D (Triangle if available; SciPy fallback).

    Returns:
      updated_vertices (N',3), updated_faces (M',3),
      insertion_counts {vertex_index: count}, mapped_indices [len(new_points)].
    """
    V = mesh.vertices.tolist()
    F = mesh.faces.astype(int)
    nV0 = len(V)

    # Projection in one shot
    new_points = np.asarray(new_points, dtype=float)
    closest, dists, fids = mesh.nearest.on_surface(new_points)

    # Precompute frames per face
    A_all, ex_all, ey_all, n_all = face_frames(V, F)

    # Dedup map (existing verts)
    key_of = lambda p: tuple(np.round(p, round_dp))
    coord_to_index = {key_of(v): i for i, v in enumerate(V)}

    insertion_counts = defaultdict(int)
    mapped_indices = [None] * len(new_points)

    # Edge bins: frozenset(i,j) -> list of (t, global_idx_placeholder, input_ids)
    edge_bins = defaultdict(list)
    # Face bins: fid -> list of (global_idx_placeholder, input_ids)
    face_bins = defaultdict(list)

    # Pass 1: classify & assign placeholders (also count duplicates against existing vertices)
    placeholder_counter = 0
    placeholder_to_global = {}  # will be filled when we allocate real verts

    for p_i, (P, fid) in enumerate(zip(closest, fids)):
        fid = int(fid)
        A, ex, ey = A_all[fid], ex_all[fid], ey_all[fid]
        i0, i1, i2 = F[fid]
        A3, B3, C3 = np.asarray(V[i0]), np.asarray(V[i1]), np.asarray(V[i2])
        u, v, w = barycentric_in_face(P, A3, B3, C3) or (1 / 3, 1 / 3, 1 / 3)

        # Vertex snap
        if u > 1 - tol:
            mapped_indices[p_i] = i0
            insertion_counts[i0] += 1
            continue
        if v > 1 - tol:
            mapped_indices[p_i] = i1
            insertion_counts[i1] += 1
            continue
        if w > 1 - tol:
            mapped_indices[p_i] = i2
            insertion_counts[i2] += 1
            continue

        # Edge snap
        if abs(u) < tol:
            # edge i1-i2, param from i1
            e0 = np.asarray(V[i2]) - np.asarray(V[i1])
            t = np.clip(
                np.dot(P - np.asarray(V[i1]), e0) / (np.dot(e0, e0) + 1e-30), 0.0, 1.0
            )
            P = (1 - t) * np.asarray(V[i1]) + t * np.asarray(V[i2])
            k = key_of(P)
            if k in coord_to_index:
                idx = coord_to_index[k]
                mapped_indices[p_i] = idx
                insertion_counts[idx] += 1
                continue
            ph = placeholder_counter
            placeholder_counter += 1
            edge_bins[frozenset((i1, i2))].append((t, ph, [p_i]))
            mapped_indices[p_i] = ph
            continue
        if abs(v) < tol:
            e0 = np.asarray(V[i0]) - np.asarray(V[i2])
            t = np.clip(
                np.dot(P - np.asarray(V[i2]), e0) / (np.dot(e0, e0) + 1e-30), 0.0, 1.0
            )
            P = (1 - t) * np.asarray(V[i2]) + t * np.asarray(V[i0])
            k = key_of(P)
            if k in coord_to_index:
                idx = coord_to_index[k]
                mapped_indices[p_i] = idx
                insertion_counts[idx] += 1
                continue
            ph = placeholder_counter
            placeholder_counter += 1
            edge_bins[frozenset((i2, i0))].append((t, ph, [p_i]))
            mapped_indices[p_i] = ph
            continue
        if abs(w) < tol:
            e0 = np.asarray(V[i1]) - np.asarray(V[i0])
            t = np.clip(
                np.dot(P - np.asarray(V[i0]), e0) / (np.dot(e0, e0) + 1e-30), 0.0, 1.0
            )
            P = (1 - t) * np.asarray(V[i0]) + t * np.asarray(V[i1])
            k = key_of(P)
            if k in coord_to_index:
                idx = coord_to_index[k]
                mapped_indices[p_i] = idx
                insertion_counts[idx] += 1
                continue
            ph = placeholder_counter
            placeholder_counter += 1
            edge_bins[frozenset((i0, i1))].append((t, ph, [p_i]))
            mapped_indices[p_i] = ph
            continue

        # Interior point for this face
        ph = placeholder_counter
        placeholder_counter += 1
        face_bins[fid].append((ph, [p_i]))
        mapped_indices[p_i] = ph

    # Pass 2: allocate real vertices for edge placeholders (sorted along edge, merge near-duplicates)
    def assign_edge_points(edge_key, chain):
        nonlocal V
        # endpoints
        i, j = tuple(edge_key)
        Pi, Pj = np.asarray(V[i]), np.asarray(V[j])
        # sort and merge by t
        chain.sort(key=lambda x: x[0])
        merged = []
        for t, ph, ids in chain:
            if merged and abs(t - merged[-1][0]) < 1e-9:
                merged[-1][2].extend(ids)
            else:
                merged.append([t, ph, ids])
        # assign indices
        for t, ph, ids in merged:
            P = (1 - t) * Pi + t * Pj
            k = key_of(P)
            if k in coord_to_index:
                gidx = coord_to_index[k]
            else:
                gidx = len(V)
                V.append(tuple(P))
                coord_to_index[k] = gidx
            placeholder_to_global[ph] = gidx
            for p_i in ids:
                mapped_indices[p_i] = gidx
                insertion_counts[gidx] += 1
        # return local order (including endpoints) for this edge for face polygon building
        return i, [placeholder_to_global[ph] for _, ph, _ in merged], j

    edge_order = {}
    for ekey, chain in edge_bins.items():
        edge_order[ekey] = assign_edge_points(ekey, chain)

    # Pass 3: allocate real vertices for interior placeholders
    for fid, items in face_bins.items():
        for ph, ids in items:
            # We can use the original projected point from 'closest'; since mapped_indices holds ph,
            # recover it via any one input id
            p_i = ids[0]
            P = closest[p_i]
            k = key_of(P)
            if k in coord_to_index:
                gidx = coord_to_index[k]
            else:
                gidx = len(V)
                V.append(tuple(P))
                coord_to_index[k] = gidx
            placeholder_to_global[ph] = gidx
            for p_i in ids:
                mapped_indices[p_i] = gidx
                insertion_counts[gidx] += 1

    # Pass 4: rebuild faces per affected face (respect edge chains)
    new_faces = []
    F_list = F.tolist()

    # Helper: local triangulation for a single face
    def triangulate_face(fid):
        i0, i1, i2 = F[fid]
        A, ex, ey, n = A_all[fid], ex_all[fid], ey_all[fid], n_all[fid]
        # Build boundary vertex sequence with edge points in order
        seq = []  # list of (global_index, local2D)

        def add_chain(i, j):
            ekey = frozenset((i, j))
            if ekey in edge_order:
                left, mids, right = edge_order[ekey]
                # orient along (i->j)
                pts = mids if left == i else list(reversed(mids))
                return [i] + pts + [j]
            else:
                return [i, j]

        boundary = []
        boundary += add_chain(i0, i1)[:-1]
        boundary += add_chain(i1, i2)[:-1]
        boundary += add_chain(i2, i0)[:-1]
        boundary.append(i0)
        # interior points for this face
        interior = [placeholder_to_global[ph] for ph, _ in face_bins.get(fid, [])]
        # Local 2D coords
        all_local_ids = []
        all_local_coords = []
        id_to_local = {}
        for g in boundary:
            if g in id_to_local:
                continue
            P = np.asarray(V[g])
            xy = to_local2(A, ex, ey, P)
            id_to_local[g] = len(all_local_ids)
            all_local_ids.append(g)
            all_local_coords.append(xy)
        for g in interior:
            if g in id_to_local:
                continue
            P = np.asarray(V[g])
            xy = to_local2(A, ex, ey, P)
            id_to_local[g] = len(all_local_ids)
            all_local_ids.append(g)
            all_local_coords.append(xy)
        P2 = np.vstack(all_local_coords)
        # segments for boundary polygon
        boundary_loc = [id_to_local[g] for g in boundary]
        segments = list(zip(boundary_loc[:-1], boundary_loc[1:]))
        # Triangulate
        if use_triangle and _HAS_TRIANGLE:
            data = {"vertices": P2, "segments": np.asarray(segments, dtype=int)}
            out = tr.triangulate(data, "pQ")  # PSLG, quiet, no Steiner points
            tris_loc = out.get("triangles", np.empty((0, 3), dtype=int))
        else:
            # Fallback: Delaunay, filter by polygon containment via barycentric wrt (i0,i1,i2)
            dela = Delaunay(P2)
            tris_loc = dela.simplices
        # Map to global
        for a, b, c in tris_loc:
            ga, gb, gc = all_local_ids[a], all_local_ids[b], all_local_ids[c]
            new_faces.append([ga, gb, gc])

    # Decide which faces to rebuild: all faces touched by inserts; otherwise keep as-is
    touched_faces = set(face_bins.keys())
    for ekey in edge_bins.keys():
        # faces that use this edge
        i, j = tuple(ekey)
        mask = np.any(F == i, axis=1) & np.any(F == j, axis=1)
        hits = np.where(mask)[0]
        touched_faces.update(hits.tolist())

    keep_mask = np.ones(len(F_list), dtype=bool)
    for fid in touched_faces:
        keep_mask[fid] = False
    # Keep untouched faces
    new_faces.extend([F_list[i] for i in range(len(F_list)) if keep_mask[i]])

    # Rebuild touched faces
    for fid in tqdm(sorted(touched_faces), desc="Triangulate faces"):
        triangulate_face(fid)

    # Cleanup
    new_faces = remove_degenerate_and_duplicate_faces(V, new_faces)
    V = np.asarray(V)
    F_new = np.asarray(new_faces, dtype=int)

    # Optional basic validity checks
    if validate:
        # Check that all faces reference valid vertex indices
        assert F_new.min() >= 0 and F_new.max() < len(V)
        # Ensure connectivity not broken (at least one component)
        assert len(F_new) > 0

    return V, F_new, dict(insertion_counts), mapped_indices


# ------------------------- geodesic utilities -------------------------


def compute_pairwise_geodesic_for_inputs(
    updated_vertices, updated_faces, mapped_indices
):
    """Return P x P geodesic matrix for the P original input points, using mapped_indices."""
    P = len(mapped_indices)
    D = np.zeros((P, P), dtype=float)
    geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)
    for i in tqdm(range(P), desc="Geodesic distances"):
        src = mapped_indices[i]
        t_subset = mapped_indices[i:]
        dists, _ = geoalg.geodesicDistances([src], t_subset)
        D[i, i:] = dists
        D[i:, i] = dists
    return D


# if __name__ == "__main__":
#     # Minimal demo: a single triangle with interior and edge inserts
#     verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
#     faces = np.array([[0, 1, 2]], int)
#     base = build_trimesh(verts, faces)

#     pts = [
#         (0.5, 0.0, 0.0),  # on edge 0-1
#         (0.25, 0.25, 0.0),  # interior
#         (0.75, 0.0, 0.0),  # on same edge 0-1
#         (0.0, 1.0, 0.0),  # snaps to existing vertex 2
#         (0.5, 0.0, 0.0),  # duplicate
#     ]

#     V2, F2, counts, mapped = insert_points_into_mesh_batch(
#         base, pts, tol=1e-8, round_dp=9, use_triangle=True, validate=True
#     )

#     D = compute_pairwise_geodesic_for_inputs(V2, F2, mapped)
#     print("mapped_indices:", mapped)
#     print("insertion_counts:", counts)
#     print("V shape:", V2.shape, "F shape:", F2.shape)


def example():
    # Minimal demo on a single triangle
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2]], int)
    base = build_trimesh(verts, faces)

    # Points include duplicates, on-edge, and interior
    pts = [
        (0.5, 0.0, 0.0),  # on edge 0-1
        (0.5, 0.0, 0.0),  # duplicate
        (0.25, 0.25, 0.0),  # interior
        (0.0, 1.0, 0.0),  # exactly existing vertex -> maps to idx 2
    ]

    V2, F2, counts, mapped = insert_points_into_mesh_batch(
        base,
        pts,
        tol=1e-8,
        round_dp=9,
        validate_each_insert=True,
        on_validation_fail="warn",
    )

    D = compute_pairwise_geodesic_for_inputs(V2, F2, mapped)
    print("mapped_indices:", mapped)
    print("insertion_counts:", counts)
    print("V shape:", V2.shape, "F shape:", F2.shape)
    print("D:\n", D)


def example_real():
    import pandas as pd
    import ast

    dataset = "jrc_22ak351-leaf-3m"
    plasmodesmata_file = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/plasmodesmata_cleaned_lines_assigned_to_2_nearest_cells.csv"
    plasmodesmata_df = pd.read_csv(plasmodesmata_file)

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
    # Read the corresponding cell CSV
    cell_file = (
        f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/cell.csv"
    )
    cell_df = pd.read_csv(cell_file)
    # Compute plasmodesmata counts per cell
    cell_df = cell_df.rename(
        columns={
            "COM X (nm)": "Cell COM X (nm)",
            "COM Y (nm)": "Cell COM Y (nm)",
            "COM Z (nm)": "Cell COM Z (nm)",
        }
    )
    # Merge the plasmodesmata counts with cell_df (using "Object ID" in cell_df)
    merged_df = cell_df.merge(
        exploded_df, left_on="Object ID", right_on="Cell ID", how="left"
    )

    # get all cells matching id
    cell_id = 100  # 390
    cell_plasmodesmata_coords = merged_df[merged_df["Cell ID"] == cell_id][
        [
            "Plasmodesmata COM X (nm)",
            "Plasmodesmata COM Y (nm)",
            "Plasmodesmata COM Z (nm)",
        ]
    ].to_numpy()

    # read in mesh
    cell_mesh_file = f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3m/cell_fixed/meshes/{cell_id}.ply"
    cell_mesh = trimesh.load_mesh(cell_mesh_file)
    cell_mesh.vertices = cell_mesh.vertices  # %%
    updated_vertices, updated_faces, insertion_counts, mapped_indices = (
        insert_points_into_mesh_batch(cell_mesh, cell_plasmodesmata_coords)
        #     , validate_each_insert=True
        # )
    )
    return updated_vertices, updated_faces, insertion_counts, mapped_indices


# updated_vertices, updated_faces, insertion_counts, mapped_indices = example_real()
# mesh = build_trimesh(updated_vertices, updated_faces)
# _ = mesh.export("example_mesh.ply")

# # %%
import pickle

data = pickle.load(
    open(
        f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3m/geodesic_distances/100_distribution.pkl",
        "rb",
    )
)
# %%
new_dist_matrix = compute_pairwise_geodesic_for_inputs(
    updated_vertices, updated_faces, mapped_indices
)
data["plasmodesmata_indices"] = mapped_indices
data["distance_matrix"] = new_dist_matrix
data["updated_vertices"] = updated_vertices
# dump new_dist_matrix to pkl file
import pickle

with open("./new_dist_matrix.pkl", "wb") as f:
    pickle.dump(data, f)
# %%


# plotting
def get_shrunk_mesh(mesh_path, factor=1):
    # assuming you already have verts, faces from cell_mesh
    mesh = trimesh.load_mesh(mesh_path, process=True, validate=True)
    mesh.remove_unreferenced_vertices()
    # mesh.compute_vertex_normals()

    normals = mesh.vertex_normals  # outward normals
    # we'll shrink *inward* by moving along -normal

    def is_inside_all(d):
        pts = verts - normals * d
        return mesh.contains(pts).all()

    # pick a safe upper bound for d (e.g. average edge length)
    avg_edge = mesh.edges_unique_length.mean()
    # lo, hi = 0.0, avg_edge * factor
    # for _ in range(20):  # 20-step binary search → sub-µm precision
    #     mid = (lo + hi) / 2
    #     if is_inside_all(mid):
    #         lo = mid
    #     else:
    #         hi = mid

    # best_d = lo
    # shrunk_verts = mesh.vertices - normals * best_d
    # print(mesh.edges_unique_length.mean(), best_d)
    shrunk_verts = mesh.vertices - normals * (mesh.edges_unique_length.mean() * factor)
    shrunk_mesh = trimesh.Trimesh(vertices=shrunk_verts, faces=mesh.faces)
    return shrunk_mesh


def compute_density(dist_matrix: np.ndarray, radius: float) -> np.ndarray:
    """
    For each row i in dist_matrix, counts how many entries
    (other than itself) are ≤ radius.

    Returns an array of shape (n,) where n = dist_matrix.shape[0].
    """
    # boolean mask where True if distance ≤ radius
    within = dist_matrix <= radius
    # sum along each row, subtract 1 to exclude self-distance==0
    return within.sum(axis=1) - 1


dataset = "jrc_22ak351-leaf-3m"
cell_id = 100
cell_mesh = trimesh.load_mesh(
    f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3m/cell_fixed/meshes/{cell_id}.ply"
)

paths = [
    f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3m/geodesic_distances/100_distribution.pkl",
    # "/nrs/cellmap/ackermand/to_delete/20250811/3m/100_distribution.pkl",
]
for i, path in enumerate(paths):
    with open(path, "rb") as f:
        data = pickle.load(f)
    dist_matrix = data["distance_matrix"]
    plasmodesmata_indices = data["plasmodesmata_indices"]
    updated_vertices = data["updated_vertices"]

    data = pickle.load(open(path, "rb"))
    plasmodesmata_indices = data["plasmodesmata_indices"]
    plasmodesmata_projected = data["updated_vertices"][plasmodesmata_indices, :]
    dist_matrix = data["distance_matrix"]
    shrunk_mesh = get_shrunk_mesh(
        f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3m/cell_fixed/meshes/{cell_id}.ply",
        factor=0.5,
    )
    if i == 1:
        plasmodesmata_projected = plasmodesmata_projected[
            :, [2, 1, 0]
        ]  # reorder for Plotly

    # example usage:
    x = 1000.0  # your chosen geodesic distance threshold
    densities = compute_density(dist_matrix, x)

    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = "vscode"

    # ---- your data here ----
    # closest = np.array([...])   # shape (N,3)
    # distance = np.array([...])  # shape (N,)
    # -------------------------

    # Scatter trace (colored by distance)
    scatter_trace = go.Scatter3d(
        x=plasmodesmata_projected[:, 0],  # updated_vertices[len(verts) :, 0],
        y=plasmodesmata_projected[:, 1],  # updated_vertices[len(verts) :, 1],
        z=plasmodesmata_projected[:, 2],  # updated_vertices[len(verts) :, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=dist_matrix[1000, :],  # densities[:],
            colorscale="Viridis",
            colorbar=dict(title="Geodesic Distance (nm)"),  # "Density"),
            opacity=0.8,
        ),
        name="samples",
    )

    # Mesh trace (semi-transparent)
    mesh_trace = go.Mesh3d(
        x=shrunk_mesh.vertices[:, 0],
        y=shrunk_mesh.vertices[:, 1],
        z=shrunk_mesh.vertices[:, 2],
        i=shrunk_mesh.faces[:, 0],  # first vertex index of each triangle
        j=shrunk_mesh.faces[:, 1],  # second
        k=shrunk_mesh.faces[:, 2],  # third
        color="gray",
        opacity=1.0,
        name="cell mesh",
    )

    fig = go.Figure(data=[mesh_trace, scatter_trace])
    fig.update_layout(
        title="Cell Mesh with Distance-colored Samples",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        autosize=True,
    )

    fig.show()

# # %%

# %%
