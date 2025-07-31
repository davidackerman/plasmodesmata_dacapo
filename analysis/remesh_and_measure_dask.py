import os
from cellmap_analyze.util import dask_util, io_util
import logging
import os
from dataclasses import dataclass
import trimesh
import pandas as pd
import ast
from remesh import insert_points_into_mesh_original
import gdist
import pygeodesic.geodesic as geodesic
import numpy as np

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


def insert_plasmodesmata_into_mesh(cell_mesh_path, cell_plasmodesmata_coords):
    """
    Insert plasmodesmata coordinates into the cell mesh.
    """
    # Load the cell mesh
    cell_mesh = trimesh.load_mesh(cell_mesh_path)
    cell_mesh.vertices = cell_mesh.vertices[
        :, ::-1
    ]  # Ensure vertices are in x,y,z order

    # Insert plasmodesmata coordinates into the mesh
    updated_vertices, updated_faces = insert_points_into_mesh_original(
        cell_mesh, cell_plasmodesmata_coords
    )
    cell_plasmodesmata_indices = list(
        range(
            len(cell_mesh.vertices),
            len(cell_mesh.vertices) + len(cell_plasmodesmata_coords),
        )
    )

    return updated_vertices, updated_faces, cell_plasmodesmata_indices


def get_geodesic_distances(vertices, faces, indices_of_interest):
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

    n = len(indices_of_interest)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        src = indices_of_interest[i]
        # only compute distances to j >= i
        target_subset = indices_of_interest[i:]
        dists, _ = geoalg.geodesicDistances([src], target_subset)
        # fill upper triangle
        dist_matrix[i, i:] = dists
        # mirror to lower triangle
        dist_matrix[i:, i] = dists

    # this could be faster:

    # d = gdist.distance_matrix_of_selected_points(
    #     updated_vertices.astype(np.float64),
    #     updated_faces.astype(np.int32),
    #     np.array(indices, dtype=np.int32),
    # ).toarray()
    # new_dist_matrix = d[-len(indices) :, -len(indices) :]
    return dist_matrix


def measure_distribution_for_cell(
    cell_plasmodesmata_coords, cell_mesh_path, output_path
):
    """
    Measure the distribution of plasmodesmata coordinates within a cell by first inserting them into the mesh.
    Then use pygeodesic to compute distances.
    """

    updated_vertices, updated_faces, cell_plasmodesmata_indices = (
        insert_plasmodesmata_into_mesh(cell_mesh_path, cell_plasmodesmata_coords)
    )
    dist_matrix = get_geodesic_distances(
        updated_vertices, updated_faces, cell_plasmodesmata_indices
    )
    cell_id = os.path.basename(cell_mesh_path).split(".")[0]
    output_file = os.path.join(output_path, f"{cell_id}_distances.npy")
    np.save(output_file, dist_matrix)


if __name__ == "__main__":
    # Initialize run properties
    run_properties = RunProperties()

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
    cell_id = 364  # 390
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
    # vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # faces = np.array([[0, 1, 2]])

    # Define new points to insert (make sure they lie in the triangle)
    new_points = np.array([[0.3, 0.3, 0.0], [0.2, 0.5, 0.0]])

    updated_vertices, updated_faces = insert_points_into_mesh_original(
        cell_mesh, cell_plasmodesmata_coords
    )
    # updated_vertices_new, updated_faces_new = insert_points_allow_duplicates(
    #     cell_mesh, cell_plasmodesmata_coords
    # )

    # new_mesh = trimesh.Trimesh(
    #     vertices=updated_vertices, faces=updated_faces, process=False
    # )
    # new_mesh.export("new_inserted.ply")
    print("Updated vertices:")
    print(updated_vertices)
    print("\nUpdated faces:")
    print(updated_faces)

    # Log the execution directory and run configuration
    logger.info(f"Execution Directory: {run_properties.execution_directory}")
    logger.info(f"Run Configuration: {run_properties.run_config}")
