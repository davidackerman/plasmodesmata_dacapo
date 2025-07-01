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


import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict


import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict
from trimesh.proximity import ProximityQuery
import trimesh
from trimesh.triangles import bounds_tree, points_to_barycentric
from trimesh.bounds import contains
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from collections import defaultdict
from tqdm import tqdm

def point_in_triangle(pt: np.ndarray, tri: np.ndarray, tol: float = 1e-8):
    """
    Compute barycentric coords of pt w.r.t. tri = [v0,v1,v2].
    Return (inside, (alpha,beta,gamma)).
    Allows small tolerance on edges/vertices.
    """
    v0, v1, v2 = tri
    u = v1 - v0
    v = v2 - v0
    w = pt - v0

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv
    # barycentric (beta, gamma) relative to v1,v2; alpha = 1 - beta - gamma
    beta  = ( uv * wv - vv * wu) / D
    gamma = ( uv * wu - uu * wv) / D
    alpha = 1 - beta - gamma

    inside = (alpha >= -tol) and (beta >= -tol) and (gamma >= -tol)
    return inside, (alpha, beta, gamma)


def build_edge_map(faces):
    """Map undirected edge (i,j) with i<j → set of face-indices containing it."""
    edge_to_faces = defaultdict(set)
    for fid, f in enumerate(faces):
        for a, b in ((0,1),(1,2),(2,0)):
            i, j = sorted((f[a], f[b]))
            edge_to_faces[(i,j)].add(fid)
    return edge_to_faces

def insert_points_allow_duplicates(mesh, new_points, tol: float = 1e-8):
    """
    Insert each point in `new_points` into `mesh` by splitting the triangle
    it falls into. Duplicates are allowed—each identical point will be inserted.
    Returns (updated_vertices, updated_faces).
    """
    # 1) Snap each new point to its nearest surface point and get its original face
    closest_pts, dists, orig_faces = mesh.nearest.on_surface(new_points)

    # 2) Work with Python lists so we can append/pop efficiently
    vertices = mesh.vertices.tolist()
    faces    = mesh.faces.tolist()

    # 3) Build mappings: original_face → set(current_face_indices), and face_idx → original_face
    n0 = len(faces)
    orig_to_current = {i: {i} for i in range(n0)}
    face_to_orig    = {i: i for i in range(n0)}

    # 4) Batch points by their original face for a tiny speedup
    pts_by_face = defaultdict(list)
    for pt, f in zip(closest_pts, orig_faces):
        pts_by_face[f].append(pt)

    # 5) Process each group
    for f_orig, pts in tqdm(pts_by_face.items(), desc="Inserting points"):
        for pt in pts:
            # 5a) Always get the up-to-date candidate faces
            candidates = orig_to_current[f_orig]
            found_fid = None

            # 5b) Try to locate which current child face contains pt
            for fid in candidates:
                tri_idxs = faces[fid]
                tri = np.array([vertices[i] for i in tri_idxs])
                if point_in_triangle(pt, tri, tol):
                    found_fid = fid
                    break

            # 5c) Fallback: if we didn’t find it geometrically, just split the first candidate
            if found_fid is None:
                # (this ensures we insert one vertex per pt)
                found_fid = next(iter(candidates))

            # 5d) Insert the new vertex
            new_vid = len(vertices)
            vertices.append(pt)

            # 5e) Split the chosen face into 3
            a, b, c = faces[found_fid]
            new_faces = [
                [a, b, new_vid],
                [b, c, new_vid],
                [c, a, new_vid],
            ]

            # 5f) Remove the old face by swapping in the last, then pop
            last_idx = len(faces) - 1
            faces[found_fid] = faces[last_idx]
            faces.pop()

            # 5g) Update mappings for the swapped‐in face (if any)
            if found_fid < last_idx:
                orig_swapped = face_to_orig[last_idx]
                face_to_orig[found_fid] = orig_swapped
                orig_to_current[orig_swapped].remove(last_idx)
                orig_to_current[orig_swapped].add(found_fid)

            # 5h) Add the 3 new faces and update mappings
            base = len(faces)
            faces.extend(new_faces)
            for i in range(3):
                fi = base + i
                face_to_orig[fi]       = f_orig
                orig_to_current[f_orig].add(fi)

            # 5i) Remove the old face index from its original mapping
            orig_to_current[f_orig].discard(found_fid)

    # 6) Return as numpy arrays
    return np.array(vertices), np.array(faces)

def insert_points_into_mesh_new_most_advanced(mesh, new_points, tol: float = 1e-8):
    """
    Inserts each point by splitting only the triangle or edge it lies on.
    Returns (new_vertices, new_faces) with exactly len(new_points) new vertices
    and no crossing triangles.
    """
    # 1) snap & get original face for each pt
    closest, dists, orig_faces = mesh.nearest.on_surface(new_points)

    # 2) mutable lists
    vertices = mesh.vertices.tolist()
    faces    = mesh.faces.tolist()

    # 3) build edge→faces mapping
    edge_to_faces = build_edge_map(faces)

    # 4) batch pts by their orig face
    pts_by_face = defaultdict(list)
    for pt, f0 in zip(closest, orig_faces):
        pts_by_face[f0].append(pt)

    # 5) process
    for f0, pts in tqdm(pts_by_face.items(), desc="Inserting points"):
        # dynamic set of current children of f0
        # (we didn’t track an orig→current map, but we know f0 itself is still there
        #  unless it’s already been split; in that case we still treat its children collectively)
        # For simplicity, just scan faces whose barycentric test succeeds—
        # there are very few of them per batch in practice.
        for pt in pts:
            # find any face that contains pt among ALL faces
            # (this is O(total_faces), but only within each batch, and each batch is small)
            found_fid = None
            found_bary = None
            for fid, f in enumerate(faces):
                tri = np.array([vertices[i] for i in f])
                inside, bary = point_in_triangle(pt, tri, tol)
                if inside:
                    found_fid, found_bary = fid, bary
                    break

            # fallback: if geometry failed, just split a random face in this batch
            if found_fid is None:
                # pick any face whose orig index was f0 originally
                # we approximate by scanning for face_to_orig info if we had it,
                # but simplest is to split the *first* face that used f0’s vertices:
                for fid, f in enumerate(faces):
                    if f0 in mesh.faces[fid]:
                        found_fid = fid
                        break
                if found_fid is None:
                    continue

            alpha, beta, gamma = found_bary

            new_vid = len(vertices)
            vertices.append(pt)

            # *** INTERIOR ***
            if (alpha > tol) and (beta > tol) and (gamma > tol):
                # split into 3
                a, b, c = faces[found_fid]
                new_tris = [[a,b,new_vid],[b,c,new_vid],[c,a,new_vid]]

                # remove old face
                last = len(faces)-1
                faces[found_fid] = faces[last]
                faces.pop()

                # update edge map for removed face
                for x,y in ((a,b),(b,c),(c,a)):
                    edge = tuple(sorted((x,y)))
                    edge_to_faces[edge].discard(last if found_fid==last else found_fid)
                    if found_fid!=last:
                        # swapped one moved into found_fid
                        edge_to_faces[edge].add(found_fid)

                # add new faces
                for tri in new_tris:
                    idx = len(faces)
                    faces.append(tri)
                    for x,y in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                        edge_to_faces[tuple(sorted((x,y)))].add(idx)

            # *** EDGE ***
            elif (alpha < tol) ^ (beta < tol) ^ (gamma < tol):
                # exactly one small → edge hit
                # identify which edge
                f = faces[found_fid]
                # pick the two indices with non-small barycentric
                vs = []
                for w,vi in zip((alpha,beta,gamma), f):
                    if w > tol:
                        vs.append(vi)
                if len(vs)!=2:
                    vs = [f[0],f[1]]  # fallback
                e0,e1 = vs
                edge = tuple(sorted((e0,e1)))
                adj = list(edge_to_faces[edge])  # faces on that edge

                # for each adjacent face, split into 2
                for fid in sorted(adj, reverse=True):
                    f = faces[fid]
                    # find orientation
                    # find where the edge sits in the face order
                    a_idx = f.index(e0)
                    b_idx = f.index(e1)
                    # ensure they are consecutive (mod 3)
                    # otherwise swap e0,e1
                    if (a_idx+1)%3 != b_idx:
                        e0,e1 = e1,e0  # swap so that e0→e1 is in the face
                        a_idx = f.index(e0)
                        b_idx = f.index(e1)
                    opp = f[3 - (a_idx + b_idx)]  # the third vertex

                    # build two new tris preserving orientation
                    t1 = [e0, new_vid, opp]
                    t2 = [new_vid, e1, opp]

                    # remove old face
                    last = len(faces)-1
                    faces[fid] = faces[last]
                    faces.pop()
                    # update edge map for removal
                    for x,y in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
                        edge_to_faces[tuple(sorted((x,y)))].discard(last if fid==last else fid)
                        if fid!=last:
                            edge_to_faces[tuple(sorted((x,y)))].add(fid)

                    # add t1,t2
                    for tri in (t1,t2):
                        idx = len(faces)
                        faces.append(tri)
                        for x,y in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                            edge_to_faces[tuple(sorted((x,y)))].add(idx)

            # *** VERTEX ***
            else:
                # pure vertex hit: append vertex only, no face changes
                # already did vertices.append(pt)
                pass

    return np.array(vertices), np.array(faces)

def find_face_on_surface(mesh, point):
    tree = bounds_tree(mesh.triangles)  # :contentReference[oaicite:0]{index=0}

    tol = 1e-6
    # 3) Make a “degenerate” AABB [min,max] = [point,point]
    query_box = np.hstack((point, point))
    # 4) O(log N) lookup of triangle indices whose AABB contains the point
    candidates = list(tree.intersection(tuple(query_box)))
    if not candidates:
        return None
    # 5) Fetch those triangles and compute barycentric coords in O(1) each
    tris = mesh.triangles[candidates]       # shape (m,3,3)
    pts = np.tile(point, (len(tris), 1))    # shape (m,3)
    bary = points_to_barycentric(tris, pts) # :contentReference[oaicite:1]{index=1}
    # 6) Check which barycentric coords lie fully inside [0,1]
    inside = np.all(bary >= -tol, axis=1) & np.all(bary <= 1+tol, axis=1)
    if np.any(inside):
        # return the first matching face index
        return candidates[np.argmax(inside)]
    return None

def insert_points_into_mesh_new(mesh: trimesh.Trimesh, new_points):
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
    # original_faces_to_updated_faces = dict(zip(faces_list, [[i] for i in faces_list]))
    for pt in tqdm(new_points):
        t0 = time.time()
        new_mesh = trimesh.Trimesh(
            vertices=vertices_list, faces=faces_list, process=False
        )
        t1 = time.time()
        face_id = find_face_on_surface(new_mesh, pt)
        t2 = time.time()
        #face_id = face_id[0]
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
        t3 = time.time()
        print(
            f"Processing point {pt} took {t3-t0:.4f}s (new mesh: {t1-t0:.4f}s, nearest search: {t2-t1:.4f}s)"
        )

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


import time
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
    # original_faces_to_updated_faces = dict(zip(faces_list, [[i] for i in faces_list]))
    for pt in tqdm(new_points):
        t0 = time.time()
        new_mesh = trimesh.Trimesh(
            vertices=vertices_list, faces=faces_list, process=False
        )
        t1 = time.time()
        _, _, face_id = new_mesh.nearest.on_surface([pt])
        t2 = time.time()
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
        t3 = time.time()
        print(
            f"Processing point {pt} took {t3-t0:.4f}s (new mesh: {t1-t0:.4f}s, nearest search: {t2-t1:.4f}s)"
        )

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


# %%
import pandas as pd
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

    # updated_vertices, updated_faces = insert_points_into_mesh_original(
    #     cell_mesh, cell_plasmodesmata_coords
    # )
    updated_vertices_new, updated_faces_new = insert_points_allow_duplicates(
        cell_mesh, cell_plasmodesmata_coords
    )

    # new_mesh = trimesh.Trimesh(
    #     vertices=updated_vertices, faces=updated_faces, process=False
    # )
    # new_mesh.export("new_inserted.ply")
    print("Updated vertices:")
    print(updated_vertices)
    print("\nUpdated faces:")
    print(updated_faces)

    # %%
    import pygeodesic.geodesic as geodesic
    import matplotlib.pyplot as plt
    id = 1000
    geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)

    distance, _ = geoalg.geodesicDistances([id], list(range(len(updated_vertices))))
    
    geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices_new, updated_faces_new)
    distance_new, _ = geoalg.geodesicDistances([id], list(range(len(updated_vertices_new))))
    for n,d,v in zip(["original","new"],[distance, distance_new],[updated_vertices,updated_vertices_new]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            v[:, 0],
            v[:, 1],
            v[:, 2],
            c=d,
            cmap="viridis",
        )
        ax.set_title(n)
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
    uv, uf = insert_points_allow_duplicates(
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
import trimesh
from trimesh.triangles import bounds_tree, points_to_barycentric
from trimesh.bounds import contains
# 2) get the R-tree on triangle AABBs

# 2) Build the R-tree on the (n,3,3) triangle array
#    This is O(N log N) and returns an rtree.Rtree index

def find_face_on_surface(mesh, point):
    tree = bounds_tree(mesh.triangles)  # :contentReference[oaicite:0]{index=0}

    tol = 1e-8
    # 3) Make a “degenerate” AABB [min,max] = [point,point]
    query_box = np.hstack((point, point))
    # 4) O(log N) lookup of triangle indices whose AABB contains the point
    candidates = list(tree.intersection(tuple(query_box)))
    if not candidates:
        return None
    # 5) Fetch those triangles and compute barycentric coords in O(1) each
    tris = mesh.triangles[candidates]       # shape (m,3,3)
    pts = np.tile(point, (len(tris), 1))    # shape (m,3)
    bary = points_to_barycentric(tris, pts) # :contentReference[oaicite:1]{index=1}
    # 6) Check which barycentric coords lie fully inside [0,1]
    inside = np.all(bary >= -tol, axis=1) & np.all(bary <= 1+tol, axis=1)
    if np.any(inside):
        # return the first matching face index
        return candidates[np.argmax(inside)]
    return None

# %%
%timeit find_face_on_surface(mesh,mesh.vertices[-1])

# %%
mesh = cell_mesh
closest_pts, dists, orig_faces = mesh.nearest.on_surface(cell_plasmodesmata_coords)

len(closest_pts), len(np.unique(closest_pts,axis=0))
# %%
