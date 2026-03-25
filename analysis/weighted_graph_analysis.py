#!/usr/bin/env python3
"""
Calculate average weighted distance between cells in a graph where edge weights
are inversely proportional to the number of plasmodesmata connections.

The weight of an edge between two cells is: 1 / #plasmodesmata
This means:
- More plasmodesmata = lower weight = easier/cheaper to traverse
- Fewer plasmodesmata = higher weight = harder/more costly to traverse

For each dataset (2l, 3m, 3r), we:
1. Build a graph with cells as nodes
2. Add edges weighted by 1/#plasmodesmata
3. Calculate shortest paths between all cell pairs
4. Analyze average weighted cost as a function of physical (Euclidean) distance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import ast
from collections import defaultdict
from scipy.spatial.distance import euclidean
import pickle
import os


def build_plasmodesmata_graph(plasmodesmata_csv, cell_csv):
    """
    Build a weighted graph where nodes are cells and edges are weighted by 1/#plasmodesmata.

    Args:
        plasmodesmata_csv: Path to plasmodesmata CSV file
        cell_csv: Path to cell CSV file

    Returns:
        G: NetworkX graph with weighted edges
        cell_positions: Dict mapping cell_id to (x, y, z) coordinates
        connection_counts: Dict mapping (cell1, cell2) tuples to plasmodesmata count
    """
    # Read CSV files
    cells_df = pd.read_csv(cell_csv)
    plasmodesmata_df = pd.read_csv(plasmodesmata_csv)

    # Create a mapping of cell ID to center of mass coordinates
    cell_positions = {}
    for idx, row in cells_df.iterrows():
        cell_id = int(row["Object ID"])
        cell_positions[cell_id] = np.array([
            row["COM X (nm)"],
            row["COM Y (nm)"],
            row["COM Z (nm)"]
        ])

    # Count plasmodesmata connections between each pair of cells
    connection_counts = defaultdict(int)

    for idx, row in plasmodesmata_df.iterrows():
        # Parse cell IDs (they're stored as strings like "[20, 28]")
        cell_ids = ast.literal_eval(row["Cell ID"])
        if len(cell_ids) == 2:
            # Create a sorted tuple to ensure consistency
            cell_pair = tuple(sorted([int(cell_ids[0]), int(cell_ids[1])]))
            connection_counts[cell_pair] += 1

    # Build the graph
    G = nx.Graph()

    # Add all cells as nodes
    for cell_id in cell_positions.keys():
        G.add_node(cell_id, pos=cell_positions[cell_id])

    # Add edges with weights = 1 / #plasmodesmata
    for (cell1, cell2), count in connection_counts.items():
        if cell1 in cell_positions and cell2 in cell_positions:
            weight = 1.0 / count  # More plasmodesmata = lower weight
            G.add_edge(cell1, cell2, weight=weight, plasmodesmata_count=count)

    return G, cell_positions, dict(connection_counts)


def calculate_euclidean_distances(cell_positions):
    """
    Calculate Euclidean distances between all pairs of cells.

    Args:
        cell_positions: Dict mapping cell_id to (x, y, z) coordinates

    Returns:
        Dict mapping (cell1, cell2) to Euclidean distance in nm
    """
    euclidean_distances = {}
    cell_ids = sorted(cell_positions.keys())

    for i, cell1 in enumerate(cell_ids):
        for cell2 in cell_ids[i+1:]:
            dist = np.linalg.norm(cell_positions[cell1] - cell_positions[cell2])
            euclidean_distances[(cell1, cell2)] = dist
            euclidean_distances[(cell2, cell1)] = dist  # symmetric

    return euclidean_distances


def calculate_shortest_path_costs(G):
    """
    Calculate shortest path costs between all pairs of connected cells using Dijkstra's algorithm.

    Args:
        G: NetworkX graph with weighted edges

    Returns:
        Dict mapping (cell1, cell2) to shortest path cost
        Dict mapping (cell1, cell2) to shortest path (list of nodes)
    """
    # Calculate all-pairs shortest paths
    path_costs = {}
    paths = {}

    # Get all node pairs in the same connected component
    for component in nx.connected_components(G):
        component_nodes = list(component)

        # Calculate shortest paths within this component
        for i, source in enumerate(component_nodes):
            # Dijkstra from this source to all other nodes in component
            lengths, paths_dict = nx.single_source_dijkstra(G, source, weight='weight')

            for target in component_nodes[i+1:]:
                if target in lengths:
                    path_costs[(source, target)] = lengths[target]
                    path_costs[(target, source)] = lengths[target]  # symmetric
                    paths[(source, target)] = paths_dict[target]
                    paths[(target, source)] = list(reversed(paths_dict[target]))

    return path_costs, paths


def analyze_cost_vs_distance(path_costs, euclidean_distances, n_bins=20):
    """
    Analyze average weighted cost as a function of Euclidean distance.

    Args:
        path_costs: Dict mapping (cell1, cell2) to shortest path cost
        euclidean_distances: Dict mapping (cell1, cell2) to Euclidean distance
        n_bins: Number of bins for grouping distances

    Returns:
        bin_centers: Array of bin centers (Euclidean distances)
        avg_costs: Array of average costs per bin
        std_costs: Array of standard deviations per bin
        counts: Array of counts per bin
    """
    # Collect pairs that have both path costs and Euclidean distances
    pairs_data = []
    for pair, cost in path_costs.items():
        if pair in euclidean_distances:
            pairs_data.append({
                'euclidean_dist': euclidean_distances[pair],
                'path_cost': cost
            })

    if not pairs_data:
        return None, None, None, None

    df = pd.DataFrame(pairs_data)

    # Bin by Euclidean distance
    bin_edges = np.linspace(df['euclidean_dist'].min(), df['euclidean_dist'].max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df['bin'] = pd.cut(df['euclidean_dist'], bins=bin_edges, labels=bin_centers, include_lowest=True)

    # Calculate statistics per bin
    grouped = df.groupby('bin', observed=False)['path_cost']
    avg_costs = grouped.mean().values
    std_costs = grouped.std().values
    counts = grouped.count().values

    return bin_centers, avg_costs, std_costs, counts


def analyze_dataset(dataset_name, plasmodesmata_csv, cell_csv, output_dir):
    """
    Perform complete analysis for a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'jrc_22ak351-leaf-2l')
        plasmodesmata_csv: Path to plasmodesmata CSV file
        cell_csv: Path to cell CSV file
        output_dir: Directory to save results

    Returns:
        Dict containing all analysis results
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # Build the graph
    print("Building plasmodesmata graph...")
    G, cell_positions, connection_counts = build_plasmodesmata_graph(
        plasmodesmata_csv, cell_csv
    )

    print(f"  Nodes (cells): {G.number_of_nodes()}")
    print(f"  Edges (cell pairs with connections): {G.number_of_edges()}")
    print(f"  Total plasmodesmata: {sum(connection_counts.values())}")

    # Calculate Euclidean distances
    print("Calculating Euclidean distances...")
    euclidean_distances = calculate_euclidean_distances(cell_positions)

    # Calculate shortest path costs
    print("Calculating shortest path costs (this may take a while)...")
    path_costs, paths = calculate_shortest_path_costs(G)

    print(f"  Cell pairs with paths: {len(path_costs) // 2}")  # Divide by 2 because symmetric

    # Analyze connected components
    components = list(nx.connected_components(G))
    print(f"  Number of connected components: {len(components)}")
    largest_component_size = max(len(c) for c in components)
    print(f"  Largest component size: {largest_component_size} cells")

    # Analyze cost vs distance
    print("Analyzing cost vs distance relationship...")
    bin_centers, avg_costs, std_costs, counts = analyze_cost_vs_distance(
        path_costs, euclidean_distances, n_bins=30
    )

    # Calculate some statistics
    if path_costs:
        avg_path_cost = np.mean(list(path_costs.values()))
        median_path_cost = np.median(list(path_costs.values()))
        max_path_cost = np.max(list(path_costs.values()))

        print(f"\nPath Cost Statistics:")
        print(f"  Average: {avg_path_cost:.4f}")
        print(f"  Median: {median_path_cost:.4f}")
        print(f"  Maximum: {max_path_cost:.4f}")

    # Edge weight statistics
    edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
    plasmodesmata_counts = [data['plasmodesmata_count'] for _, _, data in G.edges(data=True)]

    print(f"\nEdge Weight Statistics:")
    print(f"  Average edge weight: {np.mean(edge_weights):.4f}")
    print(f"  Average plasmodesmata per edge: {np.mean(plasmodesmata_counts):.2f}")
    print(f"  Max plasmodesmata per edge: {np.max(plasmodesmata_counts)}")

    # Save results
    results = {
        'dataset_name': dataset_name,
        'graph': G,
        'cell_positions': cell_positions,
        'connection_counts': connection_counts,
        'euclidean_distances': euclidean_distances,
        'path_costs': path_costs,
        'paths': paths,
        'bin_centers': bin_centers,
        'avg_costs': avg_costs,
        'std_costs': std_costs,
        'counts': counts,
        'components': components
    }

    # Save to pickle
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"{dataset_name}_weighted_graph_analysis.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")

    return results


def plot_cost_vs_distance(all_results, output_dir):
    """
    Create comparison plots of cost vs distance for all datasets.

    Args:
        all_results: Dict mapping dataset_name to results dict
        output_dir: Directory to save figures
    """
    # Create a comprehensive comparison figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    colors = {
        'jrc_22ak351-leaf-2l': 'blue',
        'jrc_22ak351-leaf-3m': 'green',
        'jrc_22ak351-leaf-3r': 'red'
    }

    # Plot 1: Average cost vs Euclidean distance (all datasets)
    ax = axes[0, 0]
    for dataset_name, results in all_results.items():
        if results['bin_centers'] is not None:
            valid = results['counts'] > 0
            ax.errorbar(
                results['bin_centers'][valid],
                results['avg_costs'][valid],
                yerr=results['std_costs'][valid] / np.sqrt(results['counts'][valid]),
                marker='o',
                linestyle='-',
                capsize=5,
                label=dataset_name.split('-')[-1],  # Just show 2l, 3m, 3r
                color=colors.get(dataset_name, 'black'),
                linewidth=2,
                markersize=6
            )
    ax.set_xlabel('Euclidean Distance (nm)', fontsize=12)
    ax.set_ylabel('Average Weighted Path Cost', fontsize=12)
    ax.set_title('Average Cost vs Euclidean Distance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Number of cell pairs per distance bin (all datasets)
    ax = axes[0, 1]
    for dataset_name, results in all_results.items():
        if results['bin_centers'] is not None:
            valid = results['counts'] > 0
            ax.plot(
                results['bin_centers'][valid],
                results['counts'][valid],
                marker='o',
                linestyle='-',
                label=dataset_name.split('-')[-1],
                color=colors.get(dataset_name, 'black'),
                linewidth=2,
                markersize=6
            )
    ax.set_xlabel('Euclidean Distance (nm)', fontsize=12)
    ax.set_ylabel('Number of Cell Pairs', fontsize=12)
    ax.set_title('Sample Size per Distance Bin', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Path cost distribution (histogram) - all datasets overlaid
    ax = axes[0, 2]
    for dataset_name, results in all_results.items():
        if results['path_costs']:
            costs = list(results['path_costs'].values())
            # Only take unique values (since dict is symmetric)
            unique_costs = [costs[i] for i in range(0, len(costs), 2)]
            ax.hist(
                unique_costs,
                bins=50,
                alpha=0.5,
                label=dataset_name.split('-')[-1],
                color=colors.get(dataset_name, 'black'),
                edgecolor='black'
            )
    ax.set_xlabel('Weighted Path Cost', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Path Costs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: All cell pairs scatter plot (all datasets)
    ax = axes[1, 0]
    for dataset_name, results in all_results.items():
        euclidean_dists = []
        path_costs_list = []
        for (c1, c2), cost in results['path_costs'].items():
            if c1 < c2:  # Only take one direction to avoid duplicates
                if (c1, c2) in results['euclidean_distances']:
                    euclidean_dists.append(results['euclidean_distances'][(c1, c2)])
                    path_costs_list.append(cost)
        ax.scatter(
            euclidean_dists,
            path_costs_list,
            alpha=0.3,
            s=5,
            color=colors.get(dataset_name, 'black'),
            label=dataset_name.split('-')[-1]
        )
    ax.set_xlabel('Euclidean Distance (nm)', fontsize=12)
    ax.set_ylabel('Weighted Path Cost', fontsize=12)
    ax.set_title('All Cell Pairs (All Datasets)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 5: Edge weight distribution (all datasets)
    ax = axes[1, 1]
    for dataset_name, results in all_results.items():
        edge_weights = [data['weight'] for _, _, data in results['graph'].edges(data=True)]
        ax.hist(
            edge_weights,
            bins=50,
            alpha=0.5,
            label=dataset_name.split('-')[-1],
            color=colors.get(dataset_name, 'black'),
            edgecolor='black'
        )
    ax.set_xlabel('Edge Weight (1 / #plasmodesmata)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Edge Weight Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 6: Plasmodesmata count distribution (all datasets)
    ax = axes[1, 2]
    for dataset_name, results in all_results.items():
        plasmodesmata_counts = [data['plasmodesmata_count'] for _, _, data in results['graph'].edges(data=True)]
        ax.hist(
            plasmodesmata_counts,
            bins=50,
            alpha=0.5,
            label=dataset_name.split('-')[-1],
            color=colors.get(dataset_name, 'black'),
            edgecolor='black'
        )
    ax.set_xlabel('Plasmodesmata Count per Connection', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Plasmodesmata Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir)
    fig_path = output_path / 'weighted_graph_analysis_comparison_all.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Combined comparison figure saved to {fig_path}")
    plt.close(fig)

    # Create individual detailed plots for each dataset
    for dataset_name, results in all_results.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Cost vs distance with error bars
        ax = axes[0, 0]
        if results['bin_centers'] is not None:
            valid = results['counts'] > 0
            ax.errorbar(
                results['bin_centers'][valid],
                results['avg_costs'][valid],
                yerr=results['std_costs'][valid] / np.sqrt(results['counts'][valid]),
                marker='o',
                linestyle='-',
                capsize=5,
                color=colors.get(dataset_name, 'blue')
            )
        ax.set_xlabel('Euclidean Distance (nm)')
        ax.set_ylabel('Average Weighted Path Cost')
        ax.set_title(f'{dataset_name}: Cost vs Distance')
        ax.grid(True, alpha=0.3)

        # Plot 2: Scatter plot of all pairs
        ax = axes[0, 1]
        euclidean_dists = []
        path_costs_list = []
        for (c1, c2), cost in results['path_costs'].items():
            if c1 < c2:  # Only take one direction to avoid duplicates
                if (c1, c2) in results['euclidean_distances']:
                    euclidean_dists.append(results['euclidean_distances'][(c1, c2)])
                    path_costs_list.append(cost)
        ax.scatter(euclidean_dists, path_costs_list, alpha=0.3, s=5, color=colors.get(dataset_name, 'blue'))
        ax.set_xlabel('Euclidean Distance (nm)')
        ax.set_ylabel('Weighted Path Cost')
        ax.set_title(f'{dataset_name}: All Cell Pairs')
        ax.grid(True, alpha=0.3)

        # Plot 3: Edge weight distribution
        ax = axes[1, 0]
        edge_weights = [data['weight'] for _, _, data in results['graph'].edges(data=True)]
        ax.hist(edge_weights, bins=50, edgecolor='black', color=colors.get(dataset_name, 'blue'))
        ax.set_xlabel('Edge Weight (1 / #plasmodesmata)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{dataset_name}: Edge Weight Distribution')
        ax.grid(True, alpha=0.3)

        # Plot 4: Plasmodesmata count distribution
        ax = axes[1, 1]
        plasmodesmata_counts = [data['plasmodesmata_count'] for _, _, data in results['graph'].edges(data=True)]
        ax.hist(plasmodesmata_counts, bins=50, edgecolor='black', color=colors.get(dataset_name, 'blue'))
        ax.set_xlabel('Plasmodesmata Count per Connection')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{dataset_name}: Plasmodesmata Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = output_path / f'{dataset_name}_detailed_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Detailed figure for {dataset_name} saved to {fig_path}")
        plt.close(fig)


def main():
    """Run complete analysis for all three datasets."""

    # Base paths
    base_results_path = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall"
    output_dir = "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/analysis/weighted_graph_results"

    # Dataset configurations
    datasets = [
        {
            "name": "jrc_22ak351-leaf-2l",
            "plasmodesmata_csv": f"{base_results_path}/jrc_22ak351-leaf-2l/plasmodesmata_lines_assigned_to_2_nearest_cells.csv",
            "cell_csv": f"{base_results_path}/jrc_22ak351-leaf-2l/cell_fixed.csv",
        },
        {
            "name": "jrc_22ak351-leaf-3m",
            "plasmodesmata_csv": f"{base_results_path}/jrc_22ak351-leaf-3m/plasmodesmata_lines_assigned_to_2_nearest_cells.csv",
            "cell_csv": f"{base_results_path}/jrc_22ak351-leaf-3m/cell_fixed.csv",
        },
        {
            "name": "jrc_22ak351-leaf-3r",
            "plasmodesmata_csv": f"{base_results_path}/jrc_22ak351-leaf-3r/plasmodesmata_lines_assigned_to_2_nearest_cells.csv",
            "cell_csv": f"{base_results_path}/jrc_22ak351-leaf-3r/cell_fixed.csv",
        },
    ]

    # Process each dataset
    all_results = {}
    for dataset in datasets:
        try:
            results = analyze_dataset(
                dataset["name"],
                dataset["plasmodesmata_csv"],
                dataset["cell_csv"],
                output_dir
            )
            all_results[dataset["name"]] = results
        except Exception as e:
            print(f"Error processing {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison plots
    if all_results:
        print(f"\n{'='*60}")
        print("Creating comparison plots...")
        print(f"{'='*60}")
        plot_cost_vs_distance(all_results, output_dir)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
