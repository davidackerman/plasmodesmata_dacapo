"""
Proposed fix for geodesic distance computation failures.

The issue: pygeodesic sometimes returns infinite distances even on connected meshes,
likely due to:
1. Numerical precision issues with very small or degenerate faces
2. Faces with near-zero area or extremely thin triangles
3. Vertices/edges with problematic geometry

The fix:
1. Validate mesh quality before geodesic computation
2. Remove or fix degenerate faces/edges
3. Add robust error handling with detailed diagnostics
4. Optionally: use graph-based fallback for failed geodesic computations
"""

import numpy as np
import trimesh
from pygeodesic import geodesic
from tqdm import tqdm


def validate_mesh_for_geodesic(vertices, faces):
    """
    Validate that a mesh is suitable for geodesic distance computation.

    Returns:
        tuple: (is_valid, issues) where issues is a list of problem descriptions
    """
    issues = []

    # Check for degenerate faces (zero area)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    face_areas = mesh.area_faces
    degenerate_faces = np.where(face_areas < 1e-10)[0]
    if len(degenerate_faces) > 0:
        issues.append(f"{len(degenerate_faces)} faces with near-zero area")

    # Check for duplicate vertices
    unique_vertices = np.unique(vertices, axis=0)
    if len(unique_vertices) < len(vertices):
        issues.append(f"{len(vertices) - len(unique_vertices)} duplicate vertices")

    # Check for very short edges
    edges = mesh.edges_unique
    edge_lengths = mesh.edges_unique_length
    very_short = np.where(edge_lengths < 1e-6)[0]
    if len(very_short) > 0:
        issues.append(f"{len(very_short)} edges with length < 1e-6")

    # Check for non-manifold edges
    if not mesh.is_watertight:
        issues.append("Mesh is not watertight")

    # Check connectivity
    if mesh.body_count > 1:
        issues.append(f"Mesh has {mesh.body_count} disconnected components")

    return len(issues) == 0, issues


def clean_mesh_for_geodesic(vertices, faces, min_area=1e-10, min_edge_length=1e-6):
    """
    Clean a mesh to improve geodesic computation reliability.

    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        min_area: Minimum face area threshold
        min_edge_length: Minimum edge length threshold

    Returns:
        tuple: (cleaned_vertices, cleaned_faces, stats)
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    stats = {
        'original_vertices': len(vertices),
        'original_faces': len(faces),
        'removed_degenerate_faces': 0,
        'merged_vertices': 0,
    }

    # Remove degenerate faces
    face_areas = mesh.area_faces
    valid_faces_mask = face_areas >= min_area
    stats['removed_degenerate_faces'] = np.sum(~valid_faces_mask)

    if stats['removed_degenerate_faces'] > 0:
        mesh.update_faces(valid_faces_mask)

    # Merge duplicate/very close vertices
    mesh.merge_vertices(digits_vertex=6)  # Merge vertices within 1e-6
    stats['merged_vertices'] = stats['original_vertices'] - len(mesh.vertices)

    stats['final_vertices'] = len(mesh.vertices)
    stats['final_faces'] = len(mesh.faces)

    return mesh.vertices, mesh.faces, stats


def compute_pairwise_geodesic_robust(
    updated_vertices, updated_faces, mapped_indices, cell_id="unknown"
):
    """
    Robustly compute pairwise geodesic distances with comprehensive error handling.

    This version:
    1. Validates and cleans the mesh first
    2. Provides detailed diagnostics on failure
    3. Attempts to identify problematic vertices
    4. Raises informative errors with actionable information
    """
    P = len(mapped_indices)

    # Step 1: Validate mesh quality
    is_valid, issues = validate_mesh_for_geodesic(updated_vertices, updated_faces)

    if not is_valid:
        print(f"[Cell {cell_id}] Mesh quality issues detected:")
        for issue in issues:
            print(f"  - {issue}")

        # Try to clean the mesh
        print(f"[Cell {cell_id}] Attempting to clean mesh...")
        cleaned_vertices, cleaned_faces, stats = clean_mesh_for_geodesic(
            updated_vertices, updated_faces
        )
        print(f"[Cell {cell_id}] Cleaning stats: {stats}")

        # Update mesh for geodesic computation
        updated_vertices = cleaned_vertices
        updated_faces = cleaned_faces

        # Re-validate
        is_valid, remaining_issues = validate_mesh_for_geodesic(updated_vertices, updated_faces)
        if not is_valid:
            raise ValueError(
                f"[Cell {cell_id}] Mesh still has quality issues after cleaning: {remaining_issues}"
            )

    # Step 2: Attempt geodesic computation
    D = np.zeros((P, P), dtype=float)

    try:
        geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)
    except Exception as e:
        raise ValueError(
            f"[Cell {cell_id}] Failed to initialize pygeodesic algorithm: {e}\n"
            f"Mesh stats: {len(updated_vertices)} vertices, {len(updated_faces)} faces"
        )

    failed_pairs = []

    for i in tqdm(range(P), desc=f"Geodesic distances (Cell {cell_id})"):
        src = mapped_indices[i]
        t_subset = mapped_indices[i:]

        try:
            dists, _ = geoalg.geodesicDistances([src], t_subset)
        except Exception as e:
            raise ValueError(
                f"[Cell {cell_id}] Geodesic computation crashed at vertex {src} (PD index {i}): {e}"
            )

        # Check for invalid distances
        invalid_mask = ~np.isfinite(dists)
        if np.any(invalid_mask):
            invalid_indices = np.where(invalid_mask)[0]
            for inv_idx in invalid_indices:
                tgt = t_subset[inv_idx]
                failed_pairs.append((src, tgt, i, i + inv_idx))

    # If we found failures, provide comprehensive diagnostics
    if failed_pairs:
        print(f"\n[Cell {cell_id}] Found {len(failed_pairs)} vertex pairs with infinite geodesic distance")
        print(f"[Cell {cell_id}] First 5 failed pairs:")
        for src_v, tgt_v, src_pd, tgt_pd in failed_pairs[:5]:
            print(f"  PD {src_pd} (vertex {src_v}) -> PD {tgt_pd} (vertex {tgt_v})")

        # Check if mesh is actually connected
        mesh = trimesh.Trimesh(vertices=updated_vertices, faces=updated_faces)
        components = mesh.split(only_watertight=False)

        if len(components) > 1:
            # Find which component each failed vertex is in
            vertex_component = {}
            for comp_idx, comp in enumerate(components):
                for v_idx, v_pos in enumerate(updated_vertices):
                    for comp_v in comp.vertices:
                        if np.allclose(v_pos, comp_v, atol=1e-9):
                            vertex_component[v_idx] = comp_idx
                            break

            print(f"\n[Cell {cell_id}] Mesh has {len(components)} disconnected components!")
            for i, comp in enumerate(components):
                print(f"  Component {i}: {len(comp.vertices)} vertices, {len(comp.faces)} faces")

            # Check which components the failed pairs are in
            src_v, tgt_v, src_pd, tgt_pd = failed_pairs[0]
            src_comp = vertex_component.get(src_v, -1)
            tgt_comp = vertex_component.get(tgt_v, -1)
            print(f"\n  Example failed pair: vertex {src_v} in component {src_comp}, vertex {tgt_v} in component {tgt_comp}")

        else:
            print(f"\n[Cell {cell_id}] Mesh appears connected ({len(components)} component), but geodesic still failing.")
            print(f"  This suggests numerical precision issues or degenerate geometry in the geodesic algorithm.")
            print(f"  Consider:")
            print(f"    1. Using a different geodesic library (e.g., gdist, igl)")
            print(f"    2. Simplifying/remeshing to improve triangle quality")
            print(f"    3. Using Euclidean distance as fallback")

        raise ValueError(
            f"[Cell {cell_id}] Geodesic distance computation failed for {len(failed_pairs)} vertex pairs. "
            f"See diagnostics above."
        )

    # Compute distance matrix (only if no failures detected)
    for i in range(P):
        src = mapped_indices[i]
        t_subset = mapped_indices[i:]
        dists, _ = geoalg.geodesicDistances([src], t_subset)
        D[i, i:] = dists
        D[i:, i] = dists

    # Final validation
    if not np.allclose(D, D.T, rtol=1e-5):
        raise ValueError(f"[Cell {cell_id}] Distance matrix is not symmetric")

    if not np.allclose(np.diag(D), 0, atol=1e-6):
        raise ValueError(f"[Cell {cell_id}] Distance matrix diagonal is not zero")

    return D


if __name__ == "__main__":
    print("This file contains the proposed fix for geodesic computation.")
    print("To apply the fix, replace compute_pairwise_geodesic_for_inputs in remesh_and_measure_dask.py")
    print("with compute_pairwise_geodesic_robust from this file.")
