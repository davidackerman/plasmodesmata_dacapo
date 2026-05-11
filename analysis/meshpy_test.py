# Surface mesh point insertion using trimesh for point projection
# %%
import numpy as np
import trimesh


# Build a trimesh surface mesh from vertices and triangular faces
def build_trimesh(vertices, faces):
    return trimesh.Trimesh(
        vertices=np.array(vertices), faces=np.array(faces), process=False
    )


# Insert points into the mesh surface by subdividing triangles
def insert_points_on_surface(vertices, faces, new_points):
    """
    Insert new_points into the mesh surface by projecting each point onto
    the current mesh and subdividing the intersecting triangle into three.

    Args:
        vertices: list of (x, y, z) tuples
        faces: list of (i0, i1, i2) index triples
        new_points: list of (x, y, z) points to insert (may be off-surface)

    Returns:
        (vertices, faces): updated lists including new vertices and faces
    """
    for pt in new_points:
        # Rebuild mesh to reflect prior insertions
        mesh = build_trimesh(vertices, faces)
        # Project the point onto the mesh surface
        closest_pts, distances, face_ids = mesh.nearest.on_surface([pt])
        closest_pt = tuple(closest_pts[0])
        face_id = face_ids[0]

        # Add the projected point as a new vertex
        new_idx = len(vertices)
        vertices.append(closest_pt)

        # Remove the original face and subdivide it
        i0, i1, i2 = faces.pop(face_id)
        faces.extend([(i0, i1, new_idx), (i1, i2, new_idx), (i2, i0, new_idx)])
    return vertices, faces


# Example usage
if __name__ == "__main__":
    # Original mesh: single triangle
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    faces = [(0, 1, 2)]

    # New points to insert (they will be projected onto the surface)
    new_points = [(0.3, 0.3, 0.1), (0.6, 0.2, -0.05)]

    vertices, faces = insert_points_on_surface(vertices, faces, new_points)
    print(f"After insertion: {len(vertices)} vertices, {len(faces)} faces")
    print("Faces:")
    for f in faces:
        print(f)

# %%
import yaml

y = yaml.load("")
