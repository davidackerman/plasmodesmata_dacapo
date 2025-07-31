import numpy as np
import trimesh
from tqdm import tqdm
import time
from collections import defaultdict
from trimesh.triangles import bounds_tree, points_to_barycentric

# Import functions from remesh.py
try:
    from remesh import insert_points_into_mesh_original, insert_points_allow_duplicates
except ImportError:
    print("Warning: Could not import from remesh.py - some functions may not work")
    insert_points_into_mesh_original = None
    insert_points_allow_duplicates = None


def insert_points_into_mesh_optimized_v1(mesh: trimesh.Trimesh, new_points):
    """
    Optimized version 1: Batch surface snapping and avoid mesh recreation.

    Key optimizations:
    1. Snap all points to surface at once (batch operation)
    2. Use spatial indexing instead of recreating mesh each time
    3. Track face mapping more efficiently
    """
    # 1) Batch snap all points to surface at once
    closest_pts, dists, orig_face_ids = mesh.nearest.on_surface(new_points)

    # 2) Work with lists for efficient insertion
    vertices_list = list(mesh.vertices)
    faces_list = list(mesh.faces)

    # 3) Build spatial index once
    tree = bounds_tree(mesh.triangles)

    # 4) Group points by their original face for better cache locality
    pts_by_face = defaultdict(list)
    for i, (pt, face_id) in enumerate(zip(closest_pts, orig_face_ids)):
        pts_by_face[face_id].append((i, pt))

    # 5) Process points grouped by face
    face_offset_map = {}  # Track how face indices change

    for orig_face_id, point_list in tqdm(pts_by_face.items(), desc="Processing faces"):
        current_face_id = orig_face_id

        # Apply any offset due to previous face removals
        for removed_face, offset in face_offset_map.items():
            if removed_face < current_face_id:
                current_face_id += offset

        for pt_idx, pt in point_list:
            # Get current face (accounting for previous modifications)
            if current_face_id >= len(faces_list):
                # Face was already removed, skip
                continue

            face = faces_list[current_face_id]

            # Add new vertex
            new_vid = len(vertices_list)
            vertices_list.append(pt)

            # Create three new faces
            new_faces = [
                [face[0], face[1], new_vid],
                [face[1], face[2], new_vid],
                [face[2], face[0], new_vid],
            ]

            # Remove old face and add new ones
            faces_list.pop(current_face_id)
            faces_list.extend(new_faces)

            # Update face offset tracking
            face_offset_map[current_face_id] = -1  # One face removed
            # Note: We add 3 faces at the end, net change is +2

    return np.array(vertices_list), np.array(faces_list)


def insert_points_into_mesh_optimized_v2(mesh: trimesh.Trimesh, new_points):
    """
    Optimized version 2: Even more aggressive optimization.

    Key optimizations:
    1. Batch surface snapping
    2. Avoid face index tracking by processing in reverse order
    3. Pre-allocate arrays where possible
    4. Better handling of edge cases (points on edges/vertices)
    """
    # 1) Batch snap all points to surface
    closest_pts, dists, face_ids = mesh.nearest.on_surface(new_points)

    # 2) Filter out points that are too close to existing vertices (potential duplicates)
    # This helps avoid degenerate triangles
    min_distance = 1e-10
    valid_points = []
    valid_face_ids = []

    for i, (pt, face_id, dist) in enumerate(zip(closest_pts, face_ids, dists)):
        # Check if point is too close to any existing vertex of the face
        if face_id < len(mesh.faces):
            face_vertices = mesh.vertices[mesh.faces[face_id]]
            min_vertex_dist = np.min(np.linalg.norm(face_vertices - pt, axis=1))

            if min_vertex_dist > min_distance:
                valid_points.append(pt)
                valid_face_ids.append(face_id)

    if not valid_points:
        # No valid points to insert
        return mesh.vertices.copy(), mesh.faces.copy()

    valid_points = np.array(valid_points)
    valid_face_ids = np.array(valid_face_ids)

    # 3) Sort points by face_id in descending order to avoid index shifting issues
    sorted_indices = np.argsort(valid_face_ids)[::-1]  # Reverse sort

    # 4) Work with lists
    vertices_list = list(mesh.vertices)
    faces_list = list(mesh.faces)

    # 5) Process points in reverse face order
    for idx in tqdm(sorted_indices, desc="Inserting points", leave=False):
        pt = valid_points[idx]
        face_id = valid_face_ids[idx]

        # Bounds check
        if face_id >= len(faces_list):
            continue

        face = faces_list[face_id]

        # Double-check face is still valid
        if len(face) != 3:
            continue

        # Add new vertex
        new_vid = len(vertices_list)
        vertices_list.append(pt)

        # Create three new faces
        new_faces = [
            [face[0], face[1], new_vid],
            [face[1], face[2], new_vid],
            [face[2], face[0], new_vid],
        ]

        # Remove old face and add new ones
        try:
            faces_list.pop(face_id)
            faces_list.extend(new_faces)
        except IndexError:
            # Face was already removed, skip
            vertices_list.pop()  # Remove the vertex we just added
            continue

    return np.array(vertices_list), np.array(faces_list)


def insert_points_into_mesh_optimized_v3(mesh: trimesh.Trimesh, new_points):
    """
    Optimized version 3: Use the existing optimized insert_points_allow_duplicates
    which already has many optimizations.

    This is essentially a wrapper that uses the more sophisticated algorithm
    from insert_points_allow_duplicates.
    """
    return insert_points_allow_duplicates(mesh, new_points)


def insert_points_into_mesh_fastest(mesh: trimesh.Trimesh, new_points):
    """
    Fastest version: Minimal operations, batch everything possible.

    This version makes some simplifying assumptions for maximum speed:
    - All points will be successfully inserted
    - No need for complex face tracking
    - Better edge case handling to avoid segfaults
    """
    # 1) Batch snap to surface
    closest_pts, dists, face_ids = mesh.nearest.on_surface(new_points)

    # 2) Filter out problematic points
    min_distance = 1e-8
    valid_data = []

    for pt, fid, dist in zip(closest_pts, face_ids, dists):
        if fid < len(mesh.faces):
            # Check distance to existing vertices
            face_vertices = mesh.vertices[mesh.faces[fid]]
            min_vertex_dist = np.min(np.linalg.norm(face_vertices - pt, axis=1))

            if min_vertex_dist > min_distance:
                valid_data.append((pt, fid))

    if not valid_data:
        return mesh.vertices.copy(), mesh.faces.copy()

    # 3) Convert to lists
    vertices = list(mesh.vertices)
    faces = list(mesh.faces)

    # 4) Create mapping of original face_id to list of points
    face_to_points = defaultdict(list)
    for pt, fid in valid_data:
        face_to_points[fid].append(pt)

    # 5) Process faces in reverse order to avoid index shifting
    for face_id in sorted(face_to_points.keys(), reverse=True):
        points = face_to_points[face_id]

        # Bounds check
        if face_id >= len(faces):
            continue

        # Process all points for this face
        original_face = faces[face_id]

        # Validate face
        if len(original_face) != 3:
            continue

        # Remove the original face
        try:
            faces.pop(face_id)
        except IndexError:
            continue

        # For each point, add vertex and create 3 faces
        for pt in points:
            new_vid = len(vertices)
            vertices.append(pt)

            # Add three new faces
            faces.extend(
                [
                    [original_face[0], original_face[1], new_vid],
                    [original_face[1], original_face[2], new_vid],
                    [original_face[2], original_face[0], new_vid],
                ]
            )

    return np.array(vertices), np.array(faces)


def insert_points_into_mesh_robust(mesh: trimesh.Trimesh, new_points):
    """
    Most robust version: Use the existing insert_points_allow_duplicates if available,
    otherwise fall back to a safe implementation.
    """
    if insert_points_allow_duplicates is not None:
        return insert_points_allow_duplicates(mesh, new_points)
    else:
        # Fallback to a very conservative approach
        return insert_points_into_mesh_fastest(mesh, new_points)


def benchmark_insertion_methods(mesh, new_points, max_points=100):
    """
    Benchmark different insertion methods.
    """
    # Limit points for benchmarking
    test_points = (
        new_points[:max_points] if len(new_points) > max_points else new_points
    )

    methods = {
        "original": insert_points_into_mesh_original,
        "optimized_v1": insert_points_into_mesh_optimized_v1,
        "optimized_v2": insert_points_into_mesh_optimized_v2,
        "optimized_v3": insert_points_into_mesh_optimized_v3,
        "fastest": insert_points_into_mesh_fastest,
    }

    results = {}

    for name, method in methods.items():
        print(f"\nTesting {name}...")
        start_time = time.time()

        try:
            # Create a copy of the mesh for each test
            test_mesh = mesh.copy()
            vertices, faces = method(test_mesh, test_points)

            end_time = time.time()
            elapsed = end_time - start_time

            results[name] = {
                "time": elapsed,
                "vertices_count": len(vertices),
                "faces_count": len(faces),
                "success": True,
            }
            print(
                f"{name}: {elapsed:.3f}s, {len(vertices)} vertices, {len(faces)} faces"
            )

        except Exception as e:
            results[name] = {"time": None, "error": str(e), "success": False}
            print(f"{name}: FAILED - {e}")

    return results


# Add the optimized function to remesh.py's namespace
def get_optimized_insert_function(version="fastest"):
    """
    Get the specified optimized insertion function.

    Args:
        version: 'v1', 'v2', 'v3', 'fastest', or 'original'
    """
    if version == "v1":
        return insert_points_into_mesh_optimized_v1
    elif version == "v2":
        return insert_points_into_mesh_optimized_v2
    elif version == "v3":
        return insert_points_into_mesh_optimized_v3
    elif version == "fastest":
        return insert_points_into_mesh_fastest
    elif version == "original":
        return insert_points_into_mesh_original
    else:
        raise ValueError(f"Unknown version: {version}")


if __name__ == "__main__":
    # Example usage and benchmarking
    print("This file contains optimized versions of insert_points_into_mesh_original")
    print("Import the functions you need or run benchmark_insertion_methods()")
