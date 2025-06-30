# %%
from collections import defaultdict
from itertools import chain, product
import numpy as np
import trimesh
import pandas as pd
import ast
from tqdm import tqdm

import fastremap
import numpy as np
from scipy.spatial import Delaunay

import numpy as np
from scipy.spatial import Delaunay
import numpy as np


def point_in_triangle_on_plane_vectorized(pt, v0, v1, v2, tol=1e-6):
    """
    Check if a single point pt lies inside many triangles defined by the arrays
    of vertices v0, v1, v2. The point is assumed to lie on the plane of each triangle.

    Parameters:
      pt : (3,) array-like, the point to test.
      v0, v1, v2 : (N, 3) array-like, vertices of the triangles.
      tol : float, tolerance for plane check and barycentric boundaries.

    Returns:
      inside : (N,) boolean numpy array where each entry is True if pt lies in
               the corresponding triangle (within tolerance), else False.
    """
    pt = np.asarray(pt).reshape(1, 3)  # shape (1, 3)
    v0 = np.asarray(v0)  # shape (N, 3)
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Step 1: Compute the normals for each triangle.
    v0v1 = v1 - v0  # shape (N, 3)
    v0v2 = v2 - v0  # shape (N, 3)
    normals = np.cross(v0v1, v0v2)  # shape (N, 3)
    norm_n = np.linalg.norm(normals, axis=1)  # shape (N,)

    # Avoid degenerate triangles.
    valid = norm_n >= tol

    # Normalize normals where possible.
    normalized_normals = np.zeros_like(normals)
    normalized_normals[valid] = normals[valid] / norm_n[valid, None]

    # Step 2: Check if the point lies in the plane of each triangle.
    # Compute the signed distance from pt to the plane of each triangle.
    # (pt - v0) will broadcast to shape (N, 3).
    distances = np.sum((pt - v0) * normalized_normals, axis=1)

    valid &= np.abs(distances) <= tol

    # Step 3: Compute barycentric coordinates to test inside/outside.
    v0pt = pt - v0  # shape (N, 3)
    dot00 = np.einsum("ij,ij->i", v0v2, v0v2)
    dot01 = np.einsum("ij,ij->i", v0v2, v0v1)
    dot02 = np.einsum("ij,ij->i", v0v2, v0pt)
    dot11 = np.einsum("ij,ij->i", v0v1, v0v1)
    dot12 = np.einsum("ij,ij->i", v0v1, v0pt)

    denom = dot00 * dot11 - dot01 * dot01
    non_degenerate = np.abs(denom) >= tol
    valid &= non_degenerate  # further mark degenerate triangles as invalid.

    invDenom = np.zeros_like(denom)
    invDenom[non_degenerate] = 1.0 / denom[non_degenerate]

    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check barycentric conditions (with tolerance).
    valid &= (u >= -tol) & (v >= -tol) & ((u + v) <= 1 + tol)

    return valid


# Example usage:
pt = [0.5, 0.5, 0.0]
# Define three triangles: one containing the point and one not.
v0 = np.array([[0, 0, 0], [0, 0, 0]])
v1 = np.array([[1, 0, 0], [1, 0, 0]])
v2 = np.array([[0, 1, 0], [1, 1, 0]])

result = point_in_triangle_on_plane_vectorized(pt, v0, v1, v2, tol=1e-6)
print(result)  # prints an array of booleans, one for each triangle


def project_points_to_plane(points, plane_origin, plane_normal):
    """
    Projects a set of 3D points onto a 2D coordinate system defined in the plane.

    Parameters:
      points (Nx3 np.array): 3D points to project.
      plane_origin (1x3 np.array): A point on the plane.
      plane_normal (1x3 np.array): Normal vector of the plane.

    Returns:
      points_2d (Nx2 np.array): The projected 2D coordinates.
      plane_x, plane_y (1x3 np.array each): The basis vectors for the plane.
    """
    # Choose an arbitrary vector that is not parallel to the normal
    arbitrary = np.array([1, 0, 0])
    if np.allclose(np.cross(plane_normal, arbitrary), 0):
        arbitrary = np.array([0, 1, 0])

    # First basis vector in the plane
    plane_x = np.cross(plane_normal, arbitrary)
    plane_x /= np.linalg.norm(plane_x)

    # Second basis vector in the plane (ensures orthogonality)
    plane_y = np.cross(plane_normal, plane_x)
    plane_y /= np.linalg.norm(plane_y)

    # Project the points: for each point, the coordinates are (dot(point - origin, plane_x), dot(point - origin, plane_y))
    relative = points - plane_origin
    u = np.dot(relative, plane_x)
    v = np.dot(relative, plane_y)
    return np.column_stack((u, v)), plane_x, plane_y


def retriangulate_planar_points(points_3d):
    """
    Retriangulates a set of 3D points (boundary plus interior) that lie on the same plane.

    This version handles duplicate vertices in the input by:
      1. Removing duplicates (keeping the first occurrence) while recording a mapping
         from each unique vertex to its duplicates in the original list.
      2. Performing Delaunay triangulation on the unique set of (projected) points.
      3. Duplicating each face as needed so that each face appears for each original
         (duplicated) vertex.

    Parameters:
      points_3d (Nx3 np.array): 3D points (which may contain duplicates) on a common plane.

    Returns:
      new_faces (Mx3 np.array): Triangles defined with indices referring to the original points.
      points_3d (Nx3 np.array): The original input vertices.
      points_2d (Nx2 np.array): The 2D projection (for the original points).
    """
    # Ensure points_3d is a NumPy array.
    points_3d = np.asarray(points_3d)

    # --- Step 1: Compress the vertices while recording duplicates ---
    # We'll keep the first occurrence of each unique vertex, comparing rows.
    unique_map = {}  # Maps a 3D coordinate (as a tuple) to its unique index.
    unique_indices = []  # List of indices in points_3d that are kept.
    inverse = []  # For each original index, the unique index it maps to.

    for i, pt in enumerate(points_3d):
        key = tuple(pt)  # Convert the point to a tuple so it can be a dict key.
        if key not in unique_map:
            unique_map[key] = len(unique_indices)
            unique_indices.append(i)
        inverse.append(unique_map[key])
    inverse = np.array(inverse)

    # unique_points: only the first instance of each vertex.
    unique_points = points_3d[unique_indices]

    # Build a mapping (dup_map) from each unique index to a list of original indices.
    dup_map = defaultdict(list)
    for orig_idx, u in enumerate(inverse):
        dup_map[u].append(orig_idx)

    # --- Step 2: Define the plane and project unique points to 2D ---
    # Use the first three unique points to define the plane.
    p0, p1, p2 = unique_points[:3]
    plane_normal = np.cross(p1 - p0, p2 - p0)
    plane_normal /= np.linalg.norm(plane_normal)

    # Project the unique points to 2D.
    points_2d_unique, plane_x, plane_y = project_points_to_plane(
        unique_points, p0, plane_normal
    )

    # Perform 2D Delaunay triangulation on the unique projected points.
    tri = Delaunay(points_2d_unique)
    simplices = tri.simplices  # Each row is a triangle (indices into unique_points)
    # --- Step 3: Expand faces to refer to the original (duplicated) vertices ---
    # For each triangle from the Delaunay triangulation (which uses unique vertex indices),
    # we generate one (or more) triangles by substituting each vertex with all possible original indices.
    all_faces = []
    print(simplices)
    for tri_unique in simplices:
        # For each vertex in the triangle, get the list of corresponding original indices.
        options = [dup_map[u] for u in tri_unique]
        vertex_ids = list(chain.from_iterable(dup_map[u] for u in tri_unique))
        if len(vertex_ids) == 3:
            all_faces.append(vertex_ids)
            continue

        # vertex_ids.remove(options[0][0])
        # if len(vertex_ids) > 3:
        #     all_faces.append((vertex_ids[-1], vertex_ids[0], vertex_ids[1]))
        # # vertex_ids.remove(options[1][0])
        # all_faces.append()
        # keep one so it is connected
        # vertex_ids.remove(options[2][0])
        duplicated_vertex_faces = list(zip(vertex_ids, vertex_ids[1:], vertex_ids[2:]))
        all_faces.extend(duplicated_vertex_faces)
        # print(options)
        # # Compute the Cartesian product of the three lists.
        # for combo in product(*options):
        #     all_faces.append(combo)
    new_faces = np.array(all_faces)
    new_faces = np.sort(new_faces, axis=1)
    new_faces = np.unique(new_faces, axis=0)  # Remove duplicate faces

    # sort new_faces by column
    # --- Step 4: Project the original points to 2D ---
    # This gives a 2D projection for the complete (duplicated) set.
    points_2d, _, _ = project_points_to_plane(points_3d, p0, plane_normal)

    return new_faces, points_3d, points_2d


def point_in_triangle_on_plane(pt, v0, v1, v2, tol=1e-6):
    """
    Check if point pt lies in the triangle defined by vertices (v0, v1, v2)
    with the condition that pt must lie on the plane of the triangle.

    Parameters:
      pt : (3,) array-like, point to test.
      v0, v1, v2 : (3,) array-like, vertices of the triangle.
      tol : float, tolerance for plane check and barycentric boundaries.

    Returns:
      bool : True if pt lies on the plane and inside the triangle (within tolerance), else False.
    """
    pt = np.array(pt)
    v0 = np.array(v0)
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Step 1: Compute the normal of the triangle's plane.
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    normal = np.cross(v0v1, v0v2)
    norm_n = np.linalg.norm(normal)
    if norm_n < tol:
        # The triangle is degenerate (area nearly zero)
        return False
    normal /= norm_n

    # Step 2: Check if the point lies in the plane.
    # Compute the signed distance from the point to the plane.
    distance = np.dot(pt - v0, normal)
    if np.abs(distance) > tol:
        # The point is not on the plane.
        return False

    # Step 3: Use barycentric coordinates to check if the point is inside the triangle.
    # Compute dot products for barycentrics.
    # Note: Use the vectors from the triangle's vertex.
    v0pt = pt - v0
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0pt)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0pt)

    denom = dot00 * dot11 - dot01 * dot01
    if np.abs(denom) < tol:
        # Degenerate triangle.
        return False
    invDenom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is inside the triangle boundaries (allowing for numerical tolerance).
    return (u >= -tol) and (v >= -tol) and (u + v <= 1 + tol)


def insert_points_into_mesh_original(mesh: trimesh.Trimesh, new_points):
    """
    Inserts new points as vertices into an existing mesh by updating the face
    list. For each new point that lies inside a triangle, the triangle is split
    into three triangles.

    Parameters:
      vertices : numpy.ndarray of shape (N, 3) representing the vertex coordinates.
      faces    : numpy.ndarray of shape (M, 3) representing each triangular face
                 as indices into the vertex array.
      new_points : numpy.ndarray of shape (P, 3) representing new points to insert.

    Returns:
      updated_vertices : numpy.ndarray, the new vertex array with inserted points.
      updated_faces    : numpy.ndarray, the updated face array after splitting.
    """
    # Convert to lists for easier insertion and removal
    closest, dists, face_id = mesh.nearest.on_surface(new_points)
    new_points = closest
    vertices_list = list(mesh.vertices)
    faces_list = list(mesh.faces)
    original_faces_to_updated_faces = dict(zip(faces_list, [[i] for i in faces_list]))
    for pt in tqdm(new_points):
        new_mesh = trimesh.Trimesh(
            vertices=vertices_list, faces=faces_list, process=False
        )
        _, _, face_id = new_mesh.nearest.on_surface([pt])
        face_id = face_id[0]
        face = new_mesh.faces[face_id]
        # Point found inside this face. Add the point to the vertex list.
        new_idx = len(vertices_list)
        vertices_list.append(pt)

        # Split the face into three new faces that include the new point.
        new_faces = [
            [face[0], face[1], new_idx],
            [face[1], face[2], new_idx],
            [face[2], face[0], new_idx],
        ]
        # print(pt, i)
        # Remove the original face and add the new faces.
        faces_list.pop(face_id)
        faces_list.extend(new_faces)
        # found_face = True
        # break  # Move on to the next new point

        # if not found_face:
        #     print(
        #         "Warning: Point", pt, "was not found in any face. It has been skipped."
        #     )

    # Convert lists back to numpy arrays for further processing
    updated_vertices = np.array(vertices_list)
    updated_faces = np.array(faces_list)
    return updated_vertices, updated_faces


def insert_points_into_mesh(mesh: trimesh.Trimesh, new_points):
    """
    Inserts new points as vertices into an existing mesh by updating the face
    list. For each new point that lies inside a triangle, the triangle is split
    into three triangles.

    Parameters:
      vertices : numpy.ndarray of shape (N, 3) representing the vertex coordinates.
      faces    : numpy.ndarray of shape (M, 3) representing each triangular face
                 as indices into the vertex array.
      new_points : numpy.ndarray of shape (P, 3) representing new points to insert.

    Returns:
      updated_vertices : numpy.ndarray, the new vertex array with inserted points.
      updated_faces    : numpy.ndarray, the updated face array after splitting.
    """
    # Convert to lists for easier insertion and removal
    closest, dists, nearest_face_ids = mesh.nearest.on_surface(new_points)

    updated_vertices = mesh.vertices
    updated_faces = mesh.faces
    # group closests by face id
    unique_face_ids = np.unique(nearest_face_ids)
    grouped_points = {fid: closest[nearest_face_ids == fid] for fid in unique_face_ids}

    # get face vertices and vstack with closest
    original_faces_to_remove = set()

    for face_id, associated_new_vertices in grouped_points.items():
        face = mesh.faces[face_id]
        face_vertices = mesh.vertices[face]
        original_faces_to_remove.add(face_id)
        # [original_vertices_to_remove.add(vertex_index) for vertex_index in face]
        combined_vertices = np.vstack((face_vertices, associated_new_vertices))
        new_faces, new_vertices, _ = retriangulate_planar_points(combined_vertices)
        new_vertices = new_vertices[3:]  # so dont include face vertices again
        new_faces -= 3
        num_vertices = len(updated_vertices)
        updated_vertices = np.vstack([updated_vertices, new_vertices])
        new_faces = new_faces + num_vertices
        fastremap.remap(
            new_faces,
            dict(zip([num_vertices - 3, num_vertices - 2, num_vertices - 1], face)),
            preserve_missing_labels=True,
            in_place=True,
        )
        updated_faces = np.vstack([updated_faces, new_faces])

    # subtract from all new ones
    # for original_vertex_to_remove in original_vertices_to_remove:
    #     updated_faces[updated_faces >= original_vertex_to_remove] -= 1
    updated_faces = np.delete(updated_faces, list(original_faces_to_remove), axis=0)
    # updated_vertices = np.delete(
    #     updated_vertices, list(original_vertices_to_remove), axis=0
    # )
    # delunay triangulate

    # delete initial face and add these faces

    # loop over faces
    # delauney for points that arent duplicates
    # new_points = closest
    # vertices_list = list(mesh.vertices)
    # faces_list = list(mesh.faces)
    # original_faces_to_updated_faces = dict(zip(faces_list, [[i] for i in faces_list]))
    # for pt in tqdm(new_points):

    #     new_mesh = trimesh.Trimesh(
    #         vertices=vertices_list, faces=faces_list, process=False
    #     )
    #     _, _, face_id = new_mesh.nearest.on_surface([pt])
    #     face_id = face_id[0]
    #     face = new_mesh.faces[face_id]
    #     # Point found inside this face. Add the point to the vertex list.
    #     new_idx = len(vertices_list)
    #     vertices_list.append(pt)

    #     # Split the face into three new faces that include the new point.
    #     new_faces = [
    #         [face[0], face[1], new_idx],
    #         [face[1], face[2], new_idx],
    #         [face[2], face[0], new_idx],
    #     ]
    #     # print(pt, i)
    #     # Remove the original face and add the new faces.
    #     faces_list.pop(face_id)
    #     faces_list.extend(new_faces)
    #     # found_face = True
    #     # break  # Move on to the next new point

    #     # if not found_face:
    #     #     print(
    #     #         "Warning: Point", pt, "was not found in any face. It has been skipped."
    #     #     )

    # # Convert lists back to numpy arrays for further processing
    # updated_vertices = np.array(vertices_list)
    # updated_faces = np.array(faces_list)
    return updated_vertices[:, ::-1], updated_faces


# uv, uf = insert_points_into_mesh(cell_mesh, cell_plasmodesmata_coords)
# trimesh.Trimesh(uv, uf, process=False).export("new_attempt.ply")
# %%

# Example usage:
if __name__ == "__main__":
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
    cell_id = 390
    cell_plasmodesmata_coords = merged_df[merged_df["Cell ID"] == cell_id][
        [
            "Plasmodesmata COM Z (nm)",
            "Plasmodesmata COM Y (nm)",
            "Plasmodesmata COM X (nm)",
        ]
    ].to_numpy()

    # read in mesh
    cell_mesh_file = f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3m/cell/meshes/{cell_id}.ply"
    cell_mesh = trimesh.load_mesh(cell_mesh_file)
    cell_mesh.vertices = cell_mesh.vertices[:, ::-1]  # vertices are in x,y,z
    num_vertices = len(cell_mesh.vertices)
    num_plasmodesmata = len(cell_plasmodesmata_coords)

    # Define a simple mesh: a single triangle
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])

    # Define new points to insert (make sure they lie in the triangle)
    new_points = np.array([[0.3, 0.3, 0.0], [0.2, 0.5, 0.0]])

    updated_vertices, updated_faces = insert_points_into_mesh_original(
        cell_mesh, cell_plasmodesmata_coords
    )

    print("Updated vertices:")
    print(updated_vertices)
    print("\nUpdated faces:")
    print(updated_faces)
    # %%
    # import numpy as np

    # def point_in_triangle_on_plane(pt, v0, v1, v2, tol=1e-6):
    #     """
    #     Check if point pt lies in the triangle defined by vertices (v0, v1, v2)
    #     with the condition that pt must lie on the plane of the triangle.

    #     Parameters:
    #       pt : (3,) array-like, point to test.
    #       v0, v1, v2 : (3,) array-like, vertices of the triangle.
    #       tol : float, tolerance for plane check and barycentric boundaries.

    #     Returns:
    #       bool : True if pt lies on the plane and inside the triangle (within tolerance), else False.
    #     """
    #     pt = np.array(pt)
    #     v0 = np.array(v0)
    #     v1 = np.array(v1)
    #     v2 = np.array(v2)

    #     # Step 1: Compute the normal of the triangle's plane.
    #     v0v1 = v1 - v0
    #     v0v2 = v2 - v0
    #     normal = np.cross(v0v1, v0v2)
    #     norm_n = np.linalg.norm(normal)
    #     if norm_n < tol:
    #         # The triangle is degenerate (area nearly zero)
    #         return False
    #     normal /= norm_n

    #     # Step 2: Check if the point lies in the plane.
    #     # Compute the signed distance from the point to the plane.
    #     distance = np.dot(pt - v0, normal)
    #     if np.abs(distance) > tol:
    #         # The point is not on the plane.
    #         return False

    #     # Step 3: Use barycentric coordinates to check if the point is inside the triangle.
    #     # Compute dot products for barycentrics.
    #     # Note: Use the vectors from the triangle's vertex.
    #     v0pt = pt - v0
    #     dot00 = np.dot(v0v2, v0v2)
    #     dot01 = np.dot(v0v2, v0v1)
    #     dot02 = np.dot(v0v2, v0pt)
    #     dot11 = np.dot(v0v1, v0v1)
    #     dot12 = np.dot(v0v1, v0pt)

    #     denom = dot00 * dot11 - dot01 * dot01
    #     if np.abs(denom) < tol:
    #         # Degenerate triangle.
    #         return False
    #     invDenom = 1.0 / denom
    #     u = (dot11 * dot02 - dot01 * dot12) * invDenom
    #     v = (dot00 * dot12 - dot01 * dot02) * invDenom

    #     # Check if point is inside the triangle boundaries (allowing for numerical tolerance).
    #     return (u >= -tol) and (v >= -tol) and (u + v <= 1 + tol)

    # point_in_triangle_on_plane(closest[0], *cell_mesh.vertices[cell_mesh.faces[831]])
    # %%
    import pygeodesic.geodesic as geodesic
    import matplotlib.pyplot as plt

    geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)

    distance, _ = geoalg.geodesicDistances([0], list(range(len(updated_vertices))))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        updated_vertices[:, 0],
        updated_vertices[:, 1],
        updated_vertices[:, 2],
        c=distance,
        cmap="viridis",
    )
    cbar = plt.colorbar(scatter)
    # %%
    import pygeodesic.geodesic as geodesic
    import matplotlib.pyplot as plt
    import numpy as np

    verts = (
        np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )  # last couple are to see about duplicates
        * 1000
    )
    faces = np.array([[0, 1, 2], [2, 3, 4], [3, 4, 5]])
    geoalg = geodesic.PyGeodesicAlgorithmExact(verts, faces)
    distance, path = geoalg.geodesicDistances([0], [0])
    print(distance, path)
    # %%
    import pygeodesic.geodesic as geodesic

    import numpy as np

    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]]
    )  # last couple are to see about duplicates
    faces = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5]])
    geoalg = geodesic.PyGeodesicAlgorithmExact(verts, faces)
    distance, path = geoalg.geodesicDistances([2], [3])
    print(distance, path)
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 0]]
    )  # last couple are to see about duplicates
    faces = np.array([[0, 1, 2], [0, 1, 4], [3, 1, 4], [3, 0, 4]])
    geoalg = geodesic.PyGeodesicAlgorithmExact(verts, faces)
    distance, path = geoalg.geodesicDistances([2], [3])
    print(distance, path)
    # %%
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay

    # Original triangle vertices
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])

    # Interior points (add as many as you like)
    internal_points = np.array([[0.3, 0.2, 0], [0.6, 0.3, 0], [0.5, 0.6, 0]])

    # Combine all points
    all_points = np.vstack((triangle, internal_points))

    # Perform Delaunay triangulation
    tri = Delaunay(all_points)

    # Plotting
    plt.triplot(all_points[:, 0], all_points[:, 1], tri.simplices)
    plt.plot(all_points[:, 0], all_points[:, 1], "o")
    plt.gca().set_aspect("equal")
    plt.title("Retriangulated Triangle")
    plt.show()

    # %%
    import numpy as np
    from scipy.spatial import Delaunay

    # Sample 3D points
    all_points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]]
    )

    # Perform Delaunay triangulation
    tri = Delaunay(all_points)

    # The 'simplices' attribute contains the indices of the vertices forming each tetrahedron
    print(tri.simplices)

    # To access the coordinates of the vertices of a specific tetrahedron:
    tetrahedron_index = 0
    vertex_indices = tri.simplices[tetrahedron_index]
    tetrahedron_vertices = all_points[vertex_indices]
    print(tetrahedron_vertices)
    # %%
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    new_points = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0.1, 0.1, 0],
    ]
    uvo, ufo = insert_points_into_mesh_original(
        trimesh.Trimesh(points, [[0, 1, 2]], process=False), new_points
    )
    uv, uf = insert_points_into_mesh(
        trimesh.Trimesh(points, [[0, 1, 2]], process=False), new_points
    )
    # uv_new, uf_new = insert_points_into_mesh(
    #     trimesh.Trimesh(points, [[0, 1, 2]], process=False),
    # )

    print(uv, uf)
    print(uvo, ufo)
    # print(uv_new, uf_new)
    # %%
    import pygeodesic.geodesic as geodesic

    geoalg = geodesic.PyGeodesicAlgorithmExact(uv, uf)
    geoalg.geodesicDistances([0], [1])
    # %%


# %%
