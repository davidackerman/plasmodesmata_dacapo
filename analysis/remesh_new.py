# %%
# Surface mesh point insertion using trimesh for point projection
# Updated to handle duplicates, edge cases, and compute geodesic distances
# Adds per-insertion geodesic validation to catch triangulation issues producing inf/NaN distances
import numpy as np
import trimesh
from collections import defaultdict
from pygeodesic import geodesic
from tqdm import tqdm
import warnings


def build_trimesh(vertices, faces):
    """Build a fresh trimesh from vertex and face lists."""
    return trimesh.Trimesh(
        vertices=np.array(vertices), faces=np.array(faces), process=False
    )


def barycentric_coords(pt, v0, v1, v2):
    """Compute barycentric coordinates for point pt in triangle v0,v1,v2."""
    M = np.column_stack((v1 - v0, v2 - v0))
    sol, *_ = np.linalg.lstsq(M, pt - v0, rcond=None)
    w, v = sol
    u = 1 - w - v
    return u, v, w


def _nonfinite_geodesics(vertices, faces, src_idx, tgt_indices):
    """Return list of (target_idx, distance) for any non-finite geodesic distances."""
    geoalg = geodesic.PyGeodesicAlgorithmExact(
        np.asarray(vertices), np.asarray(faces, dtype=int)
    )
    dists, _ = geoalg.geodesicDistances([int(src_idx)], [int(t) for t in tgt_indices])
    bad = []
    for t, d in zip(tgt_indices, dists):
        try:
            df = float(d)
        except Exception:
            df = np.inf
        if not np.isfinite(df):
            bad.append((int(t), df))
    return bad


def insert_points_into_mesh(
    mesh: trimesh.Trimesh,
    new_points,
    tol: float = 1e-8,
    validate_each_insert: bool = True,
    on_validation_fail: str = "raise",  # or "warn"
):
    """
    Inserts new_points into the mesh surface:
      - Projects with trimesh.nearest.on_surface
      - Skips duplicates (within tol) and counts them per inserted vertex
      - Splits containing triangle; splits adjacent face if point on edge
      - Records mapped index for each input point (length == len(new_points))
      - *Optionally* validates geodesic distances after each insertion from the new
        vertex to all previously inserted unique vertices; if any distance is
        inf/NaN, raises/warns immediately.

    Returns:
      updated_vertices: np.ndarray (N',3)
      updated_faces:    np.ndarray (M',3)
      insertion_counts: dict[new_vertex_index -> count]
      mapped_indices:   list[int] of length len(new_points)
    """
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    insertion_counts = defaultdict(int)
    coord_to_index = {tuple(np.round(v, 8)): idx for idx, v in enumerate(vertices)}
    mapped_indices = []
    unique_inserted = []  # track unique inserted vertex indices in order

    for pt in tqdm(new_points, desc="Insert points"):
        # Project onto surface
        closest, _, face_ids = mesh.nearest.on_surface([pt])
        proj = tuple(closest[0])
        key = tuple(np.round(proj, 8))
        # If already exists, reuse index
        if key in coord_to_index:
            idx = coord_to_index[key]
            insertion_counts[idx] += 1
            mapped_indices.append(idx)
            continue

        fid = int(face_ids[0])
        i0, i1, i2 = faces[fid]
        v0, v1, v2 = map(np.array, (vertices[i0], vertices[i1], vertices[i2]))
        u, v, w = barycentric_coords(np.array(proj), v0, v1, v2)

        # Add new vertex
        idx = len(vertices)
        vertices.append(proj)
        coord_to_index[key] = idx
        insertion_counts[idx] += 1
        mapped_indices.append(idx)
        unique_inserted.append(idx)

        # Remove containing face
        faces.pop(fid)

        # Check if point on an edge
        zero_idx = [i for i, c in enumerate((u, v, w)) if abs(c) < tol]
        if zero_idx:
            # Identify shared edge opposite the ~0 barycentric coordinate
            ei = zero_idx[0]
            if ei == 0:
                edge = {i1, i2}
            elif ei == 1:
                edge = {i2, i0}
            else:
                edge = {i0, i1}
            # Attempt to split the adjacent face along the same edge
            adj = mesh.face_adjacency
            adj_edges = mesh.face_adjacency_edges
            other_fid = None
            for j, pair in enumerate(adj):
                if fid in pair and set(adj_edges[j]) == edge:
                    other_fid = pair[0] if pair[1] == fid else pair[1]
                    break
            tris = [(i0, i1, i2)]
            if other_fid is not None:
                tris.append(tuple(mesh.faces[other_fid]))
                # Remove adjacent face as well; adjust index if needed
                faces.pop(other_fid if other_fid < fid else other_fid - 1)
            for tri in tris:
                a, b, c = tri
                shared = list(edge)
                opp = next(x for x in tri if x not in edge)
                s0, s1 = shared
                faces.extend([(s0, idx, opp), (idx, s1, opp)])
        else:
            # Strict interior: split into three
            faces.extend([(i0, i1, idx), (i1, i2, idx), (i2, i0, idx)])

        # Rebuild mesh for next steps and (optionally) validate geodesics
        mesh = build_trimesh(vertices, faces)

        if validate_each_insert and len(unique_inserted) > 1:
            # Validate distances from the newest vertex to all prior unique insertions
            prev = unique_inserted[:-1]
            bad = _nonfinite_geodesics(vertices, faces, idx, prev)
            if bad:
                msg = (
                    f"Non-finite geodesic distances after inserting vertex {idx} at {proj}. "
                    f"Problematic targets: {bad}\n"
                    f"Hint: this often indicates a disconnected surface, inverted/degenerate faces, "
                    f"or a bad edge split."
                )
                if on_validation_fail == "raise":
                    raise RuntimeError(msg)
                else:
                    warnings.warn(msg)

    updated_vertices = np.array(vertices)
    updated_faces = np.array(faces, dtype=int)
    return updated_vertices, updated_faces, dict(insertion_counts), mapped_indices


# Example usage and geodesic distance computation
def example():
    # Sample mesh
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Points to insert (with duplicates and an edge point)
    new_pts = [(0.5, 0.0, 0), (0.5, 0.0, 0), (0.3, 0.3, 0)]
    updated_vertices, updated_faces, insertion_counts, mapped_indices = (
        insert_points_into_mesh(
            mesh, new_pts, validate_each_insert=True, on_validation_fail="raise"
        )
    )

    # Compute geodesic distances among each original insertion (including duplicates mapping)
    geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)
    P = len(mapped_indices)
    dist_matrix = np.zeros((P, P), dtype=float)
    for i in tqdm(range(P), desc="Geodesic distances"):
        src = mapped_indices[i]
        targets = [mapped_indices[j] for j in range(i, P)]
        dists, _ = geoalg.geodesicDistances([src], targets)
        dist_matrix[i, i:] = dists
        dist_matrix[i:, i] = dists

    print("Pairwise geodesic distance matrix (per original points):")
    print(dist_matrix)
    print("Insertion counts (per unique inserted vertex index):")
    print(insertion_counts)


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
        insert_points_into_mesh(cell_mesh, cell_plasmodesmata_coords)
    )

example_real()
# %%
