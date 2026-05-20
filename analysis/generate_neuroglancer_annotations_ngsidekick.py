#!/usr/bin/env python3
"""
Generate precomputed Neuroglancer annotations using ngsidekick for cell centers and plasmodesmata connections.

Creates:
1. Point annotations (balls) for cell centers with volume property
2. Line annotations connecting cells with plasmodesmata, with count property

Output format is precomputed Neuroglancer annotations.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
import ast
from ngsidekick.annotations.precomputed import write_precomputed_annotations


def generate_annotations_for_dataset(
    dataset_name, plasmodesmata_csv, cell_csv, output_dir
):
    """
    Generate Neuroglancer annotations for a single dataset using ngsidekick.

    Args:
        dataset_name: Name of the dataset (e.g., 'jrc_22ak351-leaf-2l')
        plasmodesmata_csv: Path to plasmodesmata CSV file
        cell_csv: Path to cell CSV file
        output_dir: Directory to save output annotations
    """
    print(f"\nProcessing {dataset_name}...")

    # Read CSV files
    cells_df = pd.read_csv(cell_csv)
    plasmodesmata_df = pd.read_csv(plasmodesmata_csv)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate cell center annotations
    cell_data = []
    for idx, row in cells_df.iterrows():
        cell_data.append(
            {
                "x": row["COM X (nm)"],
                "y": row["COM Y (nm)"],
                "z": row["COM Z (nm)"],
                "volume": row["Volume (nm^3)"],
                "cell_id": int(row["Object ID"]),
            }
        )

    cells_annotation_df = pd.DataFrame(cell_data)
    # Convert to 32-bit types (Neuroglancer doesn't support 64-bit)
    cells_annotation_df["volume"] = cells_annotation_df["volume"].astype("float32")
    cells_annotation_df["cell_id"] = cells_annotation_df["cell_id"].astype("uint32")
    # Set index to cell IDs for annotation IDs
    cells_annotation_df.index = cells_annotation_df["cell_id"]

    # Write cell center annotations using ngsidekick
    cell_output_dir = output_path / dataset_name / "cell_centers"
    print(f"Writing {len(cells_annotation_df)} cell center annotations...")
    write_precomputed_annotations(
        df=cells_annotation_df,
        coord_space={
            "names": ["x", "y", "z"],
            "units": ["nm", "nm", "nm"],
            "scales": [1, 1, 1],
        },
        annotation_type="point",
        properties=["volume", "cell_id"],
        output_dir=str(cell_output_dir),
        description=f"Cell centers for {dataset_name}",
    )
    print(f"Saved cell center annotations to {cell_output_dir}")

    # Generate plasmodesmata connection annotations
    # Count connections between each pair of cells
    connection_counts = defaultdict(int)
    connection_endpoints = {}  # Store endpoint coordinates for each connection

    for idx, row in plasmodesmata_df.iterrows():
        # Parse cell IDs (they're stored as strings like "[20, 28]")
        cell_ids = ast.literal_eval(row["Cell ID"])
        if len(cell_ids) == 2:
            # Create a sorted tuple to ensure consistency (cell1, cell2) == (cell2, cell1)
            cell_pair = tuple(sorted([int(cell_ids[0]), int(cell_ids[1])]))
            connection_counts[cell_pair] += 1

            # Store the first occurrence's coordinates for the line
            if cell_pair not in connection_endpoints:
                # Get cell center coordinates from cells_df
                cell1_data = cells_df[cells_df["Object ID"] == cell_pair[0]]
                cell2_data = cells_df[cells_df["Object ID"] == cell_pair[1]]

                if not cell1_data.empty and not cell2_data.empty:
                    connection_endpoints[cell_pair] = {
                        "point1": [
                            cell1_data.iloc[0]["COM X (nm)"],
                            cell1_data.iloc[0]["COM Y (nm)"],
                            cell1_data.iloc[0]["COM Z (nm)"],
                        ],
                        "point2": [
                            cell2_data.iloc[0]["COM X (nm)"],
                            cell2_data.iloc[0]["COM Y (nm)"],
                            cell2_data.iloc[0]["COM Z (nm)"],
                        ],
                    }

    # Create line annotations DataFrame
    connection_data = []
    for cell_pair, count in connection_counts.items():
        if cell_pair in connection_endpoints:
            endpoints = connection_endpoints[cell_pair]
            # Use unique ID based on cell pair: cell1 * 100000 + cell2
            connection_id = cell_pair[0] * 100000 + cell_pair[1]
            connection_data.append(
                {
                    "xa": endpoints["point1"][0],
                    "ya": endpoints["point1"][1],
                    "za": endpoints["point1"][2],
                    "xb": endpoints["point2"][0],
                    "yb": endpoints["point2"][1],
                    "zb": endpoints["point2"][2],
                    "plasmodesmata_count": count,
                    "cell1_id": cell_pair[0],
                    "cell2_id": cell_pair[1],
                    "id": connection_id,
                }
            )

    connections_annotation_df = pd.DataFrame(connection_data)
    # Convert to 32-bit types (Neuroglancer doesn't support 64-bit)
    connections_annotation_df["plasmodesmata_count"] = connections_annotation_df[
        "plasmodesmata_count"
    ].astype("uint32")
    connections_annotation_df["cell1_id"] = connections_annotation_df[
        "cell1_id"
    ].astype("uint32")
    connections_annotation_df["cell2_id"] = connections_annotation_df[
        "cell2_id"
    ].astype("uint32")
    connections_annotation_df.index = connections_annotation_df["id"]
    connections_annotation_df = connections_annotation_df.drop(columns=["id"])

    # Write connection annotations using ngsidekick
    connection_output_dir = output_path / dataset_name / "plasmodesmata_connections"
    print(
        f"Writing {len(connections_annotation_df)} plasmodesmata connection annotations..."
    )
    write_precomputed_annotations(
        df=connections_annotation_df,
        coord_space={
            "names": ["x", "y", "z"],
            "units": ["nm", "nm", "nm"],
            "scales": [1, 1, 1],
        },
        annotation_type="line",
        properties=["plasmodesmata_count", "cell1_id", "cell2_id"],
        output_dir=str(connection_output_dir),
        description=f"Plasmodesmata connections for {dataset_name}",
    )
    print(f"Saved plasmodesmata connection annotations to {connection_output_dir}")

    # Print summary statistics
    total_plasmodesmata = sum(connection_counts.values())
    print(f"Summary for {dataset_name}:")
    print(f"  - Total cells: {len(cells_annotation_df)}")
    print(f"  - Total cell pairs with connections: {len(connections_annotation_df)}")
    print(f"  - Total plasmodesmata: {total_plasmodesmata}")
    if connection_counts:
        print(
            f"  - Average plasmodesmata per connection: {total_plasmodesmata / len(connection_counts):.2f}"
        )
        print(
            f"  - Max plasmodesmata in a single connection: {max(connection_counts.values())}"
        )


def main():
    """Generate precomputed annotations for all three datasets."""

    # Base paths
    base_results_path = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall"
    output_base_dir = "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/analysis/neuroglancer_annotations_precomputed_ngsidekick"

    # Dataset configurations
    datasets = [
        {
            "name": "jrc_22ak351-leaf-2l",
            "plasmodesmata_csv": f"{base_results_path}/jrc_22ak351-leaf-2l/plasmodesmata_lines_assigned_to_2_nearest_cells.csv",
            "cell_csv": f"{base_results_path}/jrc_22ak351-leaf-2l/cell_fixed.csv",
        },
        {
            "name": "jrc_22ak351-leaf-3r",
            "plasmodesmata_csv": f"{base_results_path}/jrc_22ak351-leaf-3r/plasmodesmata_lines_assigned_to_2_nearest_cells.csv",
            "cell_csv": f"{base_results_path}/jrc_22ak351-leaf-3r/cell_fixed.csv",
        },
        {
            "name": "jrc_22ak351-leaf-3m",
            "plasmodesmata_csv": f"{base_results_path}/jrc_22ak351-leaf-3m/plasmodesmata_lines_assigned_to_2_nearest_cells.csv",
            "cell_csv": f"{base_results_path}/jrc_22ak351-leaf-3m/cell_fixed.csv",
        },
    ]

    # Process each dataset
    for dataset in datasets:
        try:
            generate_annotations_for_dataset(
                dataset["name"],
                dataset["plasmodesmata_csv"],
                dataset["cell_csv"],
                output_base_dir,
            )
        except Exception as e:
            print(f"Error processing {dataset['name']}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\nAll precomputed annotations saved to: {output_base_dir}")
    print("\nTo use in Neuroglancer:")
    print("1. Add annotation layers with the 'precomputed://' source")
    print(
        f"   Example: precomputed://file://{output_base_dir}/jrc_22ak351-leaf-2l/cell_centers"
    )
    print("2. Cell centers will appear as points with volume property")
    print("3. Plasmodesmata connections will appear as lines with count property")


if __name__ == "__main__":
    main()
