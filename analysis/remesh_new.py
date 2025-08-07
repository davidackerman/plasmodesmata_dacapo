# %%
# Surface mesh point insertion using trimesh for point projection
# Robust version:
#   - Projects with trimesh.nearest.on_surface
#   - Handles duplicates (including on-vertex and on-edge cases)
#   - Splits faces with consistent orientation and exact edge snapping
#   - Removes degenerate/duplicate faces after each insert
#   - Tracks mapped_indices (one per input point) and insertion_counts
#   - Optional per-insert geodesic validation using pygeodesic

import numpy as np
import trimesh
from collections import defaultdict
from pygeodesic import geodesic
from tqdm import tqdm
import warnings

# ---------------------------- helpers ---------------------------------


def build_trimesh(vertices, faces):
    return trimesh.Trimesh(
        vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=int), process=False
    )


def face_normal(verts, tri):
    a, b, c = [np.asarray(verts[i], float) for i in tri]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 0 else n


def orient_like(verts, tri, ref_normal):
    """Flip triangle if needed so its normal has positive dot with ref_normal."""
    n = face_normal(verts, tri)
    if np.dot(n, ref_normal) < 0:
        tri = (tri[0], tri[2], tri[1])
    return tri


def barycentric_coords(P, A, B, C):
    # Robust 3D barycentric using areas
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


def edge_to_faces_map(faces):
    """Map undirected edge -> list of face indices sharing that edge."""
    m = defaultdict(list)
    for fi, (i, j, k) in enumerate(faces):
        for e in ((i, j), (j, k), (k, i)):
            m[frozenset(e)].append(fi)
    return m


def remove_faces_by_index(faces, to_remove):
    to_remove = set(to_remove)
    return [f for idx, f in enumerate(faces) if idx not in to_remove]


def remove_degenerate_and_duplicate_faces(vertices, faces, area_tol=1e-14):
    """Drop faces with repeated indices, near-zero area, or duplicates (ignoring winding)."""
    unique = {}
    cleaned = []
    V = np.asarray(vertices)
    for f in faces:
        i, j, k = f
        if i == j or j == k or k == i:
            continue
        # area check
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


def _nonfinite_geodesics(vertices, faces, src_idx, tgt_indices):
    geoalg = geodesic.PyGeodesicAlgorithmExact(
        np.asarray(vertices), np.asarray(faces, dtype=int)
    )
    dists, _ = geoalg.geodesicDistances([int(src_idx)], [int(t) for t in tgt_indices])
    bad = []
    for t, d in zip(tgt_indices, dists):
        df = float(d) if np.isfinite(d) else np.inf
        if not np.isfinite(df):
            bad.append((int(t), df))
    return bad


# ---------------------------- main API --------------------------------


def insert_points_into_mesh(
    mesh: trimesh.Trimesh,
    new_points,
    tol=1e-8,
    round_dp=9,
    validate_each_insert=False,
    on_validation_fail="raise",
):
    """
    Insert points onto a triangular surface mesh, avoiding duplicate vertices and
    splitting faces with consistent orientation. Handles on-vertex and on-edge
    cases and keeps a map from input points to final vertex indices.

    Returns:
        updated_vertices (N',3), updated_faces (M',3),
        insertion_counts {vertex_index: count}, mapped_indices [len(new_points)].
    """
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    coord_to_index = {tuple(np.round(v, round_dp)): i for i, v in enumerate(vertices)}
    insertion_counts = defaultdict(int)
    mapped_indices = []
    unique_inserted = []  # in order

    for pt in tqdm(new_points, desc="Insert points"):
        # Project to surface & find containing face
        closest_pts, _, face_ids = mesh.nearest.on_surface([pt])
        proj = np.asarray(closest_pts[0])
        fid = int(face_ids[0])

        # Check against existing vertices (including originals)
        key = tuple(np.round(proj, round_dp))
        if key in coord_to_index:
            idx = coord_to_index[key]
            insertion_counts[idx] += 1
            mapped_indices.append(idx)
            continue

        # Get containing triangle data
        i0, i1, i2 = faces[fid]
        A, B, C = (
            np.asarray(vertices[i0]),
            np.asarray(vertices[i1]),
            np.asarray(vertices[i2]),
        )
        bary = barycentric_coords(proj, A, B, C)
        if bary is None:
            # Fallback: just treat as interior to avoid crash
            bary = (1 / 3, 1 / 3, 1 / 3)
        u, v, w = bary

        # Snap to vertex if it's extremely close
        if u > 1 - tol:
            idx = i0
            insertion_counts[idx] += 1
            mapped_indices.append(idx)
            continue
        if v > 1 - tol:
            idx = i1
            insertion_counts[idx] += 1
            mapped_indices.append(idx)
            continue
        if w > 1 - tol:
            idx = i2
            insertion_counts[idx] += 1
            mapped_indices.append(idx)
            continue

        # On-edge detection; snap point onto the exact edge to avoid slivers
        on_edge = None
        if abs(u) < tol:
            on_edge = (i1, i2)
            proj = v * B + w * C  # exact edge position
        elif abs(v) < tol:
            on_edge = (i2, i0)
            proj = w * C + u * A
        elif abs(w) < tol:
            on_edge = (i0, i1)
            proj = u * A + v * B

        # Re-check duplicate after snapping
        key = tuple(np.round(proj, round_dp))
        if key in coord_to_index:
            idx = coord_to_index[key]
            insertion_counts[idx] += 1
            mapped_indices.append(idx)
            continue

        # Create new vertex
        new_idx = len(vertices)
        vertices.append(tuple(proj))
        coord_to_index[key] = new_idx
        insertion_counts[new_idx] += 1
        mapped_indices.append(new_idx)
        unique_inserted.append(new_idx)

        # Reference normal for orientation consistency
        ref_n = face_normal(vertices, (i0, i1, i2))

        # Build edge map BEFORE removing faces
        e2f = edge_to_faces_map(faces)

        # Which faces to remove and which to add
        faces_to_remove = set()
        faces_to_add = []

        if on_edge is None:
            # Strict interior: split fid into three
            faces_to_remove.add(fid)
            tris = [
                (i0, i1, new_idx),
                (i1, i2, new_idx),
                (i2, i0, new_idx),
            ]
            faces_to_add.extend([orient_like(vertices, t, ref_n) for t in tris])
        else:
            # Edge split: split both adjacent faces along the shared edge, if both exist
            edge_key = frozenset(on_edge)
            adj_faces = e2f.get(edge_key, [])
            # Remove all adj faces that use this edge (usually 1 or 2)
            for afi in adj_faces:
                faces_to_remove.add(afi)
            # For each adjacent face, split into two
            for afi in adj_faces if adj_faces else [fid]:
                a, b, c = faces[afi]
                # identify ordering relative to edge
                if frozenset((a, b)) == edge_key:
                    shared = (a, b)
                    opp = c
                elif frozenset((b, c)) == edge_key:
                    shared = (b, c)
                    opp = a
                else:
                    shared = (c, a)
                    opp = b
                s0, s1 = shared
                tris = [(s0, new_idx, opp), (new_idx, s1, opp)]
                # Use this face's normal as reference for orientation
                local_n = face_normal(vertices, (a, b, c))
                faces_to_add.extend([orient_like(vertices, t, local_n) for t in tris])

        # Apply removals/additions without index shifting issues
        faces = remove_faces_by_index(faces, faces_to_remove)
        faces.extend(faces_to_add)

        # Cleanup: drop degenerate & duplicate faces, then rebuild mesh
        faces = remove_degenerate_and_duplicate_faces(vertices, faces)
        mesh = build_trimesh(vertices, faces)

        # Optional: per-insert geodesic validation to catch topology issues early
        if validate_each_insert and len(unique_inserted) > 1:
            prev = unique_inserted[:-1]
            bad = _nonfinite_geodesics(vertices, faces, new_idx, prev)
            if bad:
                msg = (
                    f"Non-finite geodesics after inserting vertex {new_idx}: {bad}. "
                    f"Likely topology issue (disconnected component, duplicate/degenerate faces)."
                )
                if on_validation_fail == "raise":
                    raise RuntimeError(msg)
                else:
                    warnings.warn(msg)

    updated_vertices = np.asarray(vertices)
    updated_faces = np.asarray(faces, dtype=int)
    return updated_vertices, updated_faces, dict(insertion_counts), mapped_indices


# ------------------------- example & distances -------------------------


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

    V2, F2, counts, mapped = insert_points_into_mesh(
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
    cell_mesh_file = f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3m/cell/meshes/{cell_id}.ply"
    cell_mesh = trimesh.load_mesh(cell_mesh_file)
    cell_mesh.vertices = cell_mesh.vertices  # %%
    updated_vertices, updated_faces, insertion_counts, mapped_indices = (
        insert_points_into_mesh(
            cell_mesh, cell_plasmodesmata_coords, validate_each_insert=True
        )
    )
    return updated_vertices, updated_faces, insertion_counts, mapped_indices


updated_vertices, updated_faces, insertion_counts, mapped_indices = example_real()
mesh = build_trimesh(updated_vertices, updated_faces)
_ = mesh.export("example_mesh.ply")

# %%
