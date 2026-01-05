#!/usr/bin/env python3
"""
Dask-parallelized version of the remeshing and distance matrix calculation.
This script processes multiple cells in parallel using Dask.
"""

import os
import pandas as pd
import numpy as np
import ast
import trimesh
import pygeodesic.geodesic as geodesic
from pathlib import Path
import logging
from typing import Tuple, Optional, List, Dict
import pickle
from tqdm import tqdm

# Import functions from the original remesh.py
from remesh import (
    insert_points_into_mesh_original,
    insert_points_allow_duplicates,
    compute_density,
)

# Dask imports
import dask
from dask import delayed
from dask.distributed import Client, as_completed
import dask.bag as db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(dataset: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Load and prepare the plasmodesmata and cell data.

    Args:
        dataset: Dataset name (e.g., "jrc_22ak351-leaf-3m")

    Returns:
        Tuple of (merged_df, cell_file_path, mesh_dir_path)
    """
    # File paths
    plasmodesmata_file = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/plasmodesmata_cleaned_lines_assigned_to_2_nearest_cells.csv"
    cell_file = (
        f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/cell.csv"
    )
    mesh_dir = f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/{dataset}/cell/meshes"

    # Load data
    logger.info(f"Loading plasmodesmata data from {plasmodesmata_file}")
    plasmodesmata_df = pd.read_csv(plasmodesmata_file)

    # Parse list columns
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

    # Load cell data
    logger.info(f"Loading cell data from {cell_file}")
    cell_df = pd.read_csv(cell_file)
    cell_df = cell_df.rename(
        columns={
            "COM X (nm)": "Cell COM X (nm)",
            "COM Y (nm)": "Cell COM Y (nm)",
            "COM Z (nm)": "Cell COM Z (nm)",
        }
    )

    # Merge the data
    merged_df = cell_df.merge(
        exploded_df, left_on="Object ID", right_on="Cell ID", how="left"
    )

    return merged_df, cell_file, mesh_dir


def compute_geodesic_distance_matrix(
    vertices: np.ndarray, faces: np.ndarray, indices: List[int]
) -> np.ndarray:
    """
    Compute the geodesic distance matrix for the given indices on the mesh.

    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        indices: Indices of points for which to compute distances

    Returns:
        Symmetric distance matrix
    """
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

    n = len(indices)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        src = indices[i]
        # Only compute distances to j >= i (upper triangle)
        target_subset = indices[i:]
        dists, _ = geoalg.geodesicDistances([src], target_subset)
        # Fill upper triangle
        dist_matrix[i, i:] = dists
        # Mirror to lower triangle
        dist_matrix[i:, i] = dists

    return dist_matrix


@delayed
def process_single_cell(
    cell_id: int,
    merged_df: pd.DataFrame,
    mesh_dir: str,
    use_allow_duplicates: bool = True,
    save_results: bool = True,
    output_dir: str = "remesh_results",
) -> Dict:
    """
    Process a single cell: load mesh, insert plasmodesmata points, compute distance matrix.

    Args:
        cell_id: ID of the cell to process
        merged_df: Merged dataframe with cell and plasmodesmata data
        mesh_dir: Directory containing mesh files
        use_allow_duplicates: Whether to use the allow_duplicates insertion method
        save_results: Whether to save intermediate results
        output_dir: Directory to save results

    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Processing cell {cell_id}")

        # Get plasmodesmata coordinates for this cell
        cell_data = merged_df[merged_df["Cell ID"] == cell_id]
        if cell_data.empty:
            logger.warning(f"No plasmodesmata data found for cell {cell_id}")
            return {
                "cell_id": cell_id,
                "success": False,
                "error": "No plasmodesmata data found",
            }

        cell_plasmodesmata_coords = (
            cell_data[
                [
                    "Plasmodesmata COM Z (nm)",
                    "Plasmodesmata COM Y (nm)",
                    "Plasmodesmata COM X (nm)",
                ]
            ]
            .dropna()
            .to_numpy()
        )

        if len(cell_plasmodesmata_coords) == 0:
            logger.warning(f"No valid plasmodesmata coordinates for cell {cell_id}")
            return {
                "cell_id": cell_id,
                "success": False,
                "error": "No valid plasmodesmata coordinates",
            }

        # Load mesh
        cell_mesh_file = os.path.join(mesh_dir, f"{cell_id}.ply")
        if not os.path.exists(cell_mesh_file):
            logger.warning(f"Mesh file not found: {cell_mesh_file}")
            return {
                "cell_id": cell_id,
                "success": False,
                "error": f"Mesh file not found: {cell_mesh_file}",
            }

        cell_mesh = trimesh.load_mesh(cell_mesh_file)
        cell_mesh.vertices = cell_mesh.vertices[:, ::-1]  # Convert from x,y,z to z,y,x

        # Insert plasmodesmata points into mesh
        if use_allow_duplicates:
            updated_vertices, updated_faces = insert_points_allow_duplicates(
                cell_mesh, cell_plasmodesmata_coords
            )
        else:
            updated_vertices, updated_faces = insert_points_into_mesh_original(
                cell_mesh, cell_plasmodesmata_coords
            )

        # Get indices of inserted plasmodesmata points
        indices = list(
            range(
                len(cell_mesh.vertices),
                len(cell_mesh.vertices) + len(cell_plasmodesmata_coords),
            )
        )

        # Compute geodesic distance matrix
        dist_matrix = compute_geodesic_distance_matrix(
            updated_vertices, updated_faces, indices
        )

        # Compute density (example with 1000nm radius)
        densities = compute_density(dist_matrix, 1000.0)

        # Prepare results
        results = {
            "cell_id": cell_id,
            "success": True,
            "num_plasmodesmata": len(cell_plasmodesmata_coords),
            "num_original_vertices": len(cell_mesh.vertices),
            "num_updated_vertices": len(updated_vertices),
            "num_updated_faces": len(updated_faces),
            "distance_matrix": dist_matrix,
            "densities": densities,
            "plasmodesmata_coords": cell_plasmodesmata_coords,
            "updated_vertices": updated_vertices,
            "updated_faces": updated_faces,
            "plasmodesmata_indices": indices,
        }

        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            result_file = os.path.join(output_dir, f"cell_{cell_id}_results.pkl")
            with open(result_file, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Saved results for cell {cell_id} to {result_file}")

        logger.info(
            f"Successfully processed cell {cell_id}: {len(cell_plasmodesmata_coords)} plasmodesmata"
        )
        return results

    except Exception as e:
        error_msg = f"Error processing cell {cell_id}: {str(e)}"
        logger.error(error_msg)
        return {"cell_id": cell_id, "success": False, "error": error_msg}


def process_cells_parallel(
    dataset: str,
    cell_ids: Optional[List[int]] = None,
    max_workers: int = 4,
    use_allow_duplicates: bool = True,
    save_results: bool = True,
    output_dir: str = "remesh_results",
) -> List[Dict]:
    """
    Process multiple cells in parallel using Dask.

    Args:
        dataset: Dataset name
        cell_ids: List of cell IDs to process. If None, process all cells with data.
        max_workers: Maximum number of parallel workers
        use_allow_duplicates: Whether to use the allow_duplicates insertion method
        save_results: Whether to save intermediate results
        output_dir: Directory to save results

    Returns:
        List of result dictionaries
    """
    # Load and prepare data
    merged_df, cell_file, mesh_dir = load_and_prepare_data(dataset)

    # Determine which cells to process
    if cell_ids is None:
        # Get all unique cell IDs that have plasmodesmata data
        cell_ids = merged_df["Cell ID"].dropna().unique().astype(int).tolist()
        logger.info(f"Found {len(cell_ids)} cells with plasmodesmata data")

    logger.info(f"Processing {len(cell_ids)} cells using {max_workers} workers")

    # Create delayed tasks for each cell
    tasks = []
    for cell_id in cell_ids:
        task = process_single_cell(
            cell_id, merged_df, mesh_dir, use_allow_duplicates, save_results, output_dir
        )
        tasks.append(task)

    # Set up Dask client
    with Client(
        n_workers=max_workers, threads_per_worker=1, memory_limit="4GB"
    ) as client:
        logger.info(f"Dask client: {client}")

        # Execute tasks
        logger.info("Starting parallel processing...")
        results = dask.compute(*tasks)

    logger.info(f"Completed processing {len(results)} cells")

    # Print summary
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    logger.info(f"Successfully processed: {successful}, Failed: {failed}")

    if failed > 0:
        logger.info("Failed cells:")
        for r in results:
            if not r.get("success", False):
                logger.info(f"  Cell {r['cell_id']}: {r.get('error', 'Unknown error')}")

    return results


def analyze_results(
    results: List[Dict], output_dir: str = "remesh_results"
) -> pd.DataFrame:
    """
    Analyze the results and create summary statistics.

    Args:
        results: List of result dictionaries from process_cells_parallel
        output_dir: Directory to save analysis results

    Returns:
        DataFrame with summary statistics
    """
    # Extract successful results
    successful_results = [r for r in results if r.get("success", False)]

    if not successful_results:
        logger.warning("No successful results to analyze")
        return pd.DataFrame()

    # Create summary dataframe
    summary_data = []
    for r in successful_results:
        summary_data.append(
            {
                "cell_id": r["cell_id"],
                "num_plasmodesmata": r["num_plasmodesmata"],
                "num_original_vertices": r["num_original_vertices"],
                "num_updated_vertices": r["num_updated_vertices"],
                "num_updated_faces": r["num_updated_faces"],
                "mean_density": np.mean(r["densities"]),
                "std_density": np.std(r["densities"]),
                "min_density": np.min(r["densities"]),
                "max_density": np.max(r["densities"]),
                "mean_distance": np.mean(
                    r["distance_matrix"][r["distance_matrix"] > 0]
                ),
                "std_distance": np.std(r["distance_matrix"][r["distance_matrix"] > 0]),
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary statistics to {summary_file}")

    return summary_df


def main():
    """
    Main function to run the parallel processing.
    """
    # Configuration
    dataset = "jrc_22ak351-leaf-3m"

    # For testing, process only a subset of cells
    # Set to None to process all cells
    test_cell_ids = [364, 390, 400, 410, 420]  # Example cell IDs for testing

    # Process cells in parallel
    results = process_cells_parallel(
        dataset=dataset,
        cell_ids=test_cell_ids,  # Set to None for all cells
        max_workers=4,
        use_allow_duplicates=True,
        save_results=True,
        output_dir="remesh_results",
    )

    # Analyze results
    summary_df = analyze_results(results, output_dir="remesh_results")
    print("\nSummary Statistics:")
    print(summary_df)

    return results, summary_df


if __name__ == "__main__":
    results, summary_df = main()
