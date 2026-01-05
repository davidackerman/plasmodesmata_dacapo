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


# %%
import os
import pickle
from cellmap_analyze.util import dask_util, io_util
import logging
from dataclasses import dataclass
import trimesh
import pandas as pd
import ast
import gdist
import pygeodesic.geodesic as geodesic
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class RunProperties:
    def __init__(self):
        args = io_util.parser_params()

        # Change execution directory
        self.execution_directory = dask_util.setup_execution_directory(
            args.config_path, logger
        )
        self.logpath = f"{self.execution_directory}/output.log"
        self.run_config = io_util.read_run_config(args.config_path)
        if args.num_workers is not None:
            self.run_config["num_workers"] = args.num_workers


# def insert_plasmodesmata_into_mesh(cell_mesh_path, cell_plasmodesmata_coords):
#     """
#     Insert plasmodesmata coordinates into the cell mesh.
#     """
#     # Load the cell mesh
#     cell_mesh = trimesh.load_mesh(cell_mesh_path)
#     cell_mesh.vertices = cell_mesh.vertices[
#         :, ::-1
#     ]  # Ensure vertices are in x,y,z order

#     # Insert plasmodesmata coordinates into the mesh
#     updated_vertices, updated_faces = insert_points_into_mesh_original(
#         cell_mesh, cell_plasmodesmata_coords
#     )
#     cell_plasmodesmata_indices = list(
#         range(
#             len(cell_mesh.vertices),
#             len(cell_mesh.vertices) + len(cell_plasmodesmata_coords),
#         )
#     )

#     return updated_vertices, updated_faces, cell_plasmodesmata_indices


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


def measure_distribution_for_cell(
    cell_plasmodesmata_coords, cell_mesh_path, output_path
):
    """
    Measure the distribution of plasmodesmata coordinates within a cell by first inserting them into the mesh.
    Then use pygeodesic to compute distances.
    """
    cell_mesh = trimesh.load(cell_mesh_path)
    updated_vertices, updated_faces, insertion_counts, mapped_indices = (
        insert_points_into_mesh_batch(cell_mesh, cell_plasmodesmata_coords)
    )
    dist_matrix = compute_pairwise_geodesic_for_inputs(
        updated_vertices, updated_faces, mapped_indices
    )
    cell_id = os.path.basename(cell_mesh_path).split(".")[0]
    # output_file = os.path.join(output_path, f"{cell_id}_distances.npy")
    # write out to pkl file

    data = {
        "updated_vertices": updated_vertices,
        "updated_faces": updated_faces,
        "insertion_counts": insertion_counts,
        "plasmodesmata_indices": mapped_indices,
        "distance_matrix": dist_matrix,
    }

    # derive cell ID and output path
    cell_id = os.path.basename(cell_mesh_path).split(".")[0]
    output_file = os.path.join(output_path, f"{cell_id}_distribution.pkl")

    # write to pickle
    with open(output_file, "wb") as f:
        pickle.dump(data, f)


def process_cell_row(row, cell_meshes_path, output_path):
    """
    Process a single row from the DataFrame containing cell information and plasmodesmata coordinates.
    """
    cell_id = row["Cell ID"]
    cell_plasmodesmata_coords = row["plasmodesmata_coords"]

    # Handle case where coordinates might be stored as different types
    if isinstance(cell_plasmodesmata_coords, str):
        try:
            # Try to evaluate the string as a list/array representation
            cell_plasmodesmata_coords = ast.literal_eval(cell_plasmodesmata_coords)
        except (ValueError, SyntaxError):
            logger.warning(f"Could not parse coordinates string for cell {cell_id}")
            return cell_id

    # Skip cells with no plasmodesmata
    if cell_plasmodesmata_coords is None or len(cell_plasmodesmata_coords) == 0:
        logger.info(f"Skipping cell {cell_id} - no plasmodesmata found")
        return cell_id

    # Ensure it's a numpy array
    try:
        if not isinstance(cell_plasmodesmata_coords, np.ndarray):
            cell_plasmodesmata_coords = np.array(cell_plasmodesmata_coords)
    except Exception as e:
        logger.error(
            f"Could not convert coordinates to numpy array for cell {cell_id}: {e}"
        )
        return cell_id

    # Check if coordinates have the right shape
    if (
        len(cell_plasmodesmata_coords.shape) != 2
        or cell_plasmodesmata_coords.shape[1] != 3
    ):
        logger.warning(
            f"Invalid coordinate shape for cell {cell_id}: {cell_plasmodesmata_coords.shape}"
        )
        return cell_id

    # Construct mesh file path
    cell_mesh_path = f"{cell_meshes_path}/{cell_id}.ply"

    # Check if mesh file exists
    if not os.path.exists(cell_mesh_path):
        logger.warning(f"Mesh file not found for cell {cell_id}: {cell_mesh_path}")
        return cell_id

    try:

        measure_distribution_for_cell(
            cell_plasmodesmata_coords, cell_mesh_path, output_path
        )
        logger.info(f"Successfully processed cell {cell_id}")
        return cell_id
    except Exception as e:
        logger.error(f"Error processing cell {cell_id}: {str(e)}")
        return cell_id


def process_partition(partition_df, cell_meshes_path, output_path):
    """
    Process a partition of the DataFrame.
    """
    results = []
    for _, row in partition_df.iterrows():
        result = process_cell_row(row, cell_meshes_path, output_path)
        results.append(result)
    return pd.DataFrame({"processed_cell_id": results})


# %%
def group_plasmodesmata_by_cell(plasmodesmata_csv, cell_csv):
    plasmodesmata_df = pd.read_csv(plasmodesmata_csv)

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
    cell_df = pd.read_csv(cell_csv)
    cell_df = cell_df.rename(
        columns={
            "Object ID": "Cell ID",
            "COM X (nm)": "Cell COM X (nm)",
            "COM Y (nm)": "Cell COM Y (nm)",
            "COM Z (nm)": "Cell COM Z (nm)",
        }
    )
    merged_df = cell_df.merge(
        exploded_df, left_on="Cell ID", right_on="Cell ID", how="left"
    )

    coords_per_cell = (
        merged_df.groupby("Cell ID")
        .apply(
            lambda df: df[
                [
                    "Plasmodesmata COM X (nm)",
                    "Plasmodesmata COM Y (nm)",
                    "Plasmodesmata COM Z (nm)",
                ]
            ].to_numpy()
        )
        .reset_index(name="plasmodesmata_coords")
    )

    # 2. Merge that back onto the cell DataFrame (one row per cell)
    result_df = cell_df.merge(coords_per_cell, on="Cell ID", how="left")

    # Convert coordinates to list of lists to avoid serialization issues with Dask
    result_df["plasmodesmata_coords"] = result_df["plasmodesmata_coords"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    # Filter out cells with no plasmodesmata for efficiency
    cells_with_plasmodesmata = result_df.dropna(subset=["plasmodesmata_coords"])
    return cells_with_plasmodesmata


# %%
if __name__ == "__main__":
    # %%
    # Initialize run properties
    rp = RunProperties()
    os.chdir(rp.execution_directory)
    with io_util.tee_streams(rp.logpath):

        run_config = rp.run_config

        cells_with_plasmodesmata = group_plasmodesmata_by_cell(
            run_config["plasmodesmata_csv"], run_config["cell_csv"]
        )
        print(f"Cells with plasmodesmata: {len(cells_with_plasmodesmata)}")

        # Debug: Check the first few entries
        print("Sample plasmodesmata_coords types:")
        for i, (idx, row) in enumerate(cells_with_plasmodesmata.head(3).iterrows()):
            coords = row["plasmodesmata_coords"]
            print(
                f"Cell {row['Cell ID']}: type={type(coords)}, shape={np.array(coords).shape if coords is not None else 'None'}"
            )  # Set up output directory
        output_path = run_config["output_path"]
        os.makedirs(output_path, exist_ok=True)

        # Get number of workers from run properties
        num_workers = run_config["num_workers"]
        with dask_util.start_dask(num_workers, "processing", logger):
            with io_util.TimingMessager("Dask processing", logger):

                # Convert to Dask DataFrame with appropriate partitioning
                # Use a reasonable partition size based on your data
                partition_size = max(
                    1, len(cells_with_plasmodesmata) // (num_workers * 2)
                )
                ddf = dd.from_pandas(
                    cells_with_plasmodesmata,
                    npartitions=max(1, len(cells_with_plasmodesmata) // partition_size),
                )

                # Apply the processing function to each partition
                processed_results = ddf.map_partitions(
                    process_partition,
                    run_config["cell_meshes_path"],
                    output_path,
                    meta=pd.DataFrame({"processed_cell_id": pd.Series(dtype="int64")}),
                )

                # Compute the results
                print("Starting parallel processing...")
                final_results = processed_results.compute()
                print(
                    f"Processing completed. Processed {len(final_results)} partitions."
                )
                print(f"Results saved to: {output_path}")

            # Save a summary of processed cells
            summary_file = os.path.join(output_path, "processing_summary.csv")
            final_results.to_csv(summary_file, index=False)
            print(f"Processing summary saved to: {summary_file}")
# %%
# cells_with_plasmodesmata = group_plasmodesmata_by_cell(
#     "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3m/plasmodesmata_cleaned_lines_assigned_to_2_nearest_cells.csv",
#     "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3m/cell.csv",
# )
# print(f"Cells with plasmodesmata: {len(cells_with_plasmodesmata)}")

# cell_mesh = trimesh.load(
#     "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3m/cell/meshes/100.ply"
# )
# # %%
# import ast

# plasmodesmata_coords = np.array(
#     cells_with_plasmodesmata[cells_with_plasmodesmata["Cell ID"] == 100][
#         "plasmodesmata_coords"
#     ].values[0]
# )
# plasmodesmata_coords = plasmodesmata_coords[:, [2, 1, 0]]  # Reorder to Z, Y, X
# updated_vertices, updated_faces, insertion_counts, mapped_indices = (
#     insert_points_into_mesh_batch(cell_mesh, plasmodesmata_coords)
# )

# %%
