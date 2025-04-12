# %%
import numpy as np
import trimesh
import pandas as pd
import ast
from tqdm import tqdm


import numpy as np


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
    closest, dists, face_id = mesh.nearest.on_surface(new_points)
    new_points = closest
    vertices_list = list(mesh.vertices)
    faces_list = list(mesh.faces)

    for pt in tqdm(new_points):
        found_face = False
        # Test each face to see if the point lies within it.
        # for i, face in enumerate(faces_list):
        #     v0 = np.array(vertices_list[face[0]])
        #     v1 = np.array(vertices_list[face[1]])
        #     v2 = np.array(vertices_list[face[2]])

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

    updated_vertices, updated_faces = insert_points_into_mesh(
        cell_mesh, cell_plasmodesmata_coords
    )

    print("Updated vertices:")
    print(updated_vertices)
    print("\nUpdated faces:")
    print(updated_faces)
# %%
# 144,831
closest, dists, face_id = trimesh.proximity.closest_point(
    cell_mesh, cell_plasmodesmata_coords
)
point_in_triangle(closest[0], *cell_mesh.vertices[cell_mesh.faces[831]])

# %%
import numpy as np


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


point_in_triangle_on_plane(closest[0], *cell_mesh.vertices[cell_mesh.faces[831]])
# %%
import pygeodesic.geodesic as geodesic
import matplotlib.pyplot as plt

more_points = trimesh.Trimesh(updated_vertices, updated_faces, process=False)
geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)

distance, _ = geoalg.geodesicDistances([len(updated_vertices)-400], list(range(len(updated_vertices))))
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
verts = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]) * 1000
faces = np.array([[0, 1, 2], [1, 3, 2]])
geoalg = geodesic.PyGeodesicAlgorithmExact(verts, faces)
distance, path = geoalg.geodesicDistances([0], [3])
print(distance, path)
# %%
