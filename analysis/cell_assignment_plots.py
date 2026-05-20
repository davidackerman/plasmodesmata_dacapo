# %%
# individual datasets
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# ===== CONFIGURATION =====
# Set to True to use existing CSV/PKL files for plotting without rerunning analysis
# When True, the script will:
#   - Load previously computed results from pickle files in DATA_CACHE_DIR
#   - Skip the time-consuming data processing and analysis steps
#   - Only regenerate plots/figures from cached data
# When False (default), the script will:
#   - Run the full analysis pipeline
#   - Process raw CSV files
#   - Save intermediate results to DATA_CACHE_DIR for future use
#
# Cache files generated:
#   - all_merged_dfs.pkl: Merged cell and plasmodesmata data with counts/densities
#   - vector_dfs.pkl: Vector categorization data (X/Y/Z orientation analysis)
#   - rdf_results.pkl: Radial distribution function results
#   - rdf_results_regions.pkl: RDF results split by anatomical regions
USE_EXISTING_DATA = True

# Directory to save/load processed data
DATA_CACHE_DIR = "/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/analysis/cached_data"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
# =========================

# List of datasets to process
datasets = ["jrc_22ak351-leaf-2l", "jrc_22ak351-leaf-3m", "jrc_22ak351-leaf-3r"]

# Dictionary to store the merged dataframe for each dataset (for combined plots)
all_merged_dfs = {}

# Define the COM coordinates and the quantities to plot
coordinates = ["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]
quantities = {
    "Plasmodesmata Count": {
        "col": "Plasmodesmata Count",
        "ylabel": "Plasmodesmata Count Per Cell",
        "ylim": (0, 5000),
    },
    "Plasmodesmata Density": {
        "col": "Plasmodesmata Density",
        "ylabel": "Plasmodesmata Density Per Cell (per nm\u00b2)",
        "ylim": (0, 3e-6),
    },
}

# Number of bins for grouping in the combined analysis
n_bins = 20

# -----------------------------------------------------------
# First: Process each dataset and generate individual scatter plots.
# -----------------------------------------------------------
if USE_EXISTING_DATA:
    # Load existing processed data
    print("Loading existing processed data from cache...")
    cache_file = os.path.join(DATA_CACHE_DIR, "all_merged_dfs.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            all_merged_dfs = pickle.load(f)
        print(f"Loaded data for {len(all_merged_dfs)} datasets from {cache_file}")
    else:
        print(
            f"WARNING: Cache file not found at {cache_file}. Running analysis instead."
        )
        USE_EXISTING_DATA = False

if not USE_EXISTING_DATA:
    # Run the full analysis
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # Read the plasmodesmata CSV and convert list-like string columns to actual lists
        plasmodesmata_file = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/plasmodesmata_cleaned_lines_assigned_to_2_nearest_cells.csv"
        plasmodesmata_df = pd.read_csv(plasmodesmata_file)
        for list_column in ["Cell ID", "Cell Distance (nm)"]:
            plasmodesmata_df[list_column] = plasmodesmata_df[list_column].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        # Explode "Cell ID" so each cell appears in its own row
        exploded_df = plasmodesmata_df.explode("Cell ID")

        # Compute plasmodesmata counts per cell
        plasmodesmata_count = exploded_df["Cell ID"].value_counts().reset_index()
        plasmodesmata_count.columns = ["Cell ID", "Plasmodesmata Count"]

        # Read the corresponding cell CSV
        cell_file = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/cell.csv"
        cell_df = pd.read_csv(cell_file)

        # Merge the plasmodesmata counts with cell_df (using "Object ID" in cell_df)
        merged_df = cell_df.merge(
            plasmodesmata_count, left_on="Object ID", right_on="Cell ID", how="left"
        )
        merged_df["Plasmodesmata Count"] = merged_df["Plasmodesmata Count"].fillna(0)

        # Compute Plasmodesmata density as Plasmodesmata Count / Surface Area (nm^2)
        merged_df["Plasmodesmata Density"] = (
            merged_df["Plasmodesmata Count"] / merged_df["Surface Area (nm^2)"]
        )

        # Filter only cells with volume greater than 1E9 nm^3
        merged_df = merged_df[merged_df["Volume (nm^3)"] > 1e9]

        # Store processed dataframe for combined plotting
        all_merged_dfs[dataset] = merged_df.copy()

    # Save processed data for future use
    cache_file = os.path.join(DATA_CACHE_DIR, "all_merged_dfs.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(all_merged_dfs, f)
    print(f"Saved processed data to {cache_file}")

# Generate scatter plots for each quantity vs. COM coordinates for each dataset.
for dataset in datasets:
    merged_df = all_merged_dfs[dataset]
    for quantity, props in quantities.items():
        fig, axs = plt.subplots(1, len(coordinates), figsize=(18, 5))
        for i, coord in enumerate(coordinates):
            axs[i].scatter(merged_df[coord], merged_df[props["col"]], alpha=0.7)
            axs[i].set_xlabel(f"Cell {coord}")
            axs[i].set_ylabel(props["ylabel"])
            axs[i].set_title(f"{quantity} vs {coord}")
            if props["ylim"]:
                axs[i].set_ylim(*props["ylim"])
        plt.suptitle(f"{quantity} for Dataset: {dataset}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(
            f'/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/analysis/figures/{dataset}_{quantity.replace(" ", "_").lower()}',
            dpi=300,
        )  # Save to PNG file, high resolution (300 dpi)
        plt.show()

# -----------------------------------------------------------
# Second: Produce combined line plots with error bars (as subplots) for each quantity.
# One figure per quantity, with one subplot for each COM coordinate.
# -----------------------------------------------------------
for quantity, props in quantities.items():
    fig, axs = plt.subplots(1, len(coordinates), figsize=(18, 5))
    for i, coord in enumerate(coordinates):
        ax = axs[i]
        # Determine common bin edges for this coordinate across all datasets
        all_coord_values = np.concatenate(
            [all_merged_dfs[ds][coord].values for ds in datasets]
        )
        bin_edges = np.linspace(
            all_coord_values.min(), all_coord_values.max(), n_bins + 1
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # For each dataset, bin the data and compute the mean and standard error
        for ds in datasets:
            df = all_merged_dfs[ds].copy()
            # Bin the coordinate data; using bin_centers as labels for convenience.
            df["bin"] = pd.cut(
                df[coord], bins=bin_edges, labels=bin_centers, include_lowest=True
            )

            # Group by binned coordinate and compute mean and SEM for the quantity.
            # Pass observed=False to silence the future warning.
            group = df.groupby("bin", observed=False)[props["col"]]
            mean_vals = group.mean()
            std_vals = group.std()
            count_vals = group.count()
            sem_vals = std_vals / np.sqrt(count_vals)

            # Only consider bins that have at least one data point
            valid = count_vals > 0
            ax.errorbar(
                mean_vals.index.astype(float)[valid],
                mean_vals.values[valid],
                yerr=sem_vals.values[valid],
                marker="o",
                linestyle="-",
                capsize=5,
                label=ds,
            )
        ax.set_xlabel(coord)
        ax.set_ylabel(props["ylabel"])
        ax.set_title(f"{quantity} vs {coord}")
        if props["ylim"]:
            ax.set_ylim(*props["ylim"])
        ax.legend()
    plt.suptitle(f"{quantity} (Combined Datasets)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        f'/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/analysis/figures/all_datasets_{quantity.replace(" ", "_").lower()}',
        dpi=300,
    )  # Save to PNG file, high resolution (300 dpi)
    plt.show()

# %%
# alignment
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# PARAMETERS & DATASET LIST
# ---------------------------
datasets = ["jrc_22ak351-leaf-2l", "jrc_22ak351-leaf-3m", "jrc_22ak351-leaf-3r"]
n_bins = 20  # for binning along cell coordinates

# We will store two kinds of processed DataFrames per dataset:
# 1. merged_df: as before (cell data with Plasmodesmata count/density; filtered by volume)
# 2. vector_df: per cell vector-categorization counts and fractions (computed from plasmodesmata connections)

all_merged_dfs = {}  # used previously for Plasmodesmata count/density plots (if needed)
vector_dfs = (
    {}
)  # will store, for each cell (per dataset), the counts and fraction of plasmodesmata vector orientations

# Define COM coordinates to be used for binning/plotting
com_coords = ["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]

# ---------------------------
# PROCESS EACH DATASET
# ---------------------------
if USE_EXISTING_DATA:
    # Load existing vector data
    print("Loading existing vector data from cache...")
    vector_cache_file = os.path.join(DATA_CACHE_DIR, "vector_dfs.pkl")
    if os.path.exists(vector_cache_file):
        with open(vector_cache_file, "rb") as f:
            vector_dfs = pickle.load(f)
        print(
            f"Loaded vector data for {len(vector_dfs)} datasets from {vector_cache_file}"
        )
    else:
        print(
            f"WARNING: Vector cache file not found at {vector_cache_file}. Running analysis instead."
        )
        USE_EXISTING_DATA = False

if not USE_EXISTING_DATA:
    # Run the full vector analysis
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # ---- Read plasmodesmata file and convert list-like columns ----
        plasmodesmata_file = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/plasmodesmata_cleaned_lines_assigned_to_2_nearest_cells.csv"
        plasmodesmata_df = pd.read_csv(plasmodesmata_file)
        # Convert columns containing list data
        for col in ["Cell ID", "Cell Distance (nm)"]:
            plasmodesmata_df[col] = plasmodesmata_df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        # ---- Explode for Plasmodesmata Count (as before) ----
        exploded_df = plasmodesmata_df.explode("Cell ID")
        plasmodesmata_count = exploded_df["Cell ID"].value_counts().reset_index()
        plasmodesmata_count.columns = ["Cell ID", "Plasmodesmata Count"]

        # ---- Read corresponding cell file ----
        cell_file = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/cell.csv"
        cell_df = pd.read_csv(cell_file)

        # Merge Plasmodesmata counts with cell data.
        merged_df = cell_df.merge(
            plasmodesmata_count, left_on="Object ID", right_on="Cell ID", how="left"
        )
        merged_df["Plasmodesmata Count"] = merged_df["Plasmodesmata Count"].fillna(0)

        # Compute Plasmodesmata Density
        merged_df["Plasmodesmata Density"] = (
            merged_df["Plasmodesmata Count"] / merged_df["Surface Area (nm^2)"]
        )

        # ---- Filter: only cells with Volume > 1E9 nm^3 ----
        merged_df = merged_df[merged_df["Volume (nm^3)"] > 1e9].copy()

        # Store the merged_df for possible further plots
        all_merged_dfs[dataset] = merged_df.copy()

        # ---------------------------
        # VECTOR CATEGORIZATION CALCULATION BASED ON Plasmodesmata LINE COORDINATES
        # ---------------------------
        # For each plasmodesmata, use its line start/end coordinates to calculate the vector.
        # Then, assign the resulting category ("Z", "X" or "Y") to both connected cells.
        vec_records = []  # will accumulate one record per cell "vote"

        # Get set of valid cells (those in merged_df) to restrict our analysis.
        valid_cells = set(merged_df["Object ID"].unique())

        for idx, row in plasmodesmata_df.iterrows():
            cell_ids = row["Cell ID"]  # list of two unique cell ids
            if not (isinstance(cell_ids, list) and len(cell_ids) == 2):
                continue
            # Sort the pair for consistency (order does not matter)
            pair = sorted(cell_ids)
            # Require that both cells pass the volume filter (are in valid_cells)
            if pair[0] not in valid_cells or pair[1] not in valid_cells:
                continue

            # Calculate vector components using plasmodesmata line coordinates.
            # It is assumed that the following columns exist in plasmodesmata_df:
            # "Line Start Z (nm)", "Line End Z (nm)", "Line Start Y (nm)", "Line End Y (nm)",
            # "Line Start X (nm)", "Line End X (nm)"
            dz = row["Line End Z (nm)"] - row["Line Start Z (nm)"]
            dy = row["Line End Y (nm)"] - row["Line Start Y (nm)"]
            dx = row["Line End X (nm)"] - row["Line Start X (nm)"]

            # Determine dominant axis based on the absolute magnitude of each component.
            if abs(dz) >= abs(dx) and abs(dz) >= abs(dy):
                cat = "Z"
            elif abs(dx) >= abs(dz) and abs(dx) >= abs(dy):
                cat = "X"
            else:
                cat = "Y"

            # Record a vote for both cells in the connection.
            vec_records.append({"Object ID": pair[0], "category": cat})
            vec_records.append({"Object ID": pair[1], "category": cat})

        # Build a DataFrame from the votes.
        vector_df = pd.DataFrame(vec_records)

        # Group by cell and count votes per category.
        if not vector_df.empty:
            cat_counts = (
                vector_df.groupby("Object ID")["category"]
                .value_counts()
                .unstack(fill_value=0)
            )
        else:
            cat_counts = pd.DataFrame(columns=["X", "Y", "Z"])
        cat_counts.reset_index(inplace=True)

        # Ensure all category columns are present.
        for col in ["X", "Y", "Z"]:
            if col not in cat_counts.columns:
                cat_counts[col] = 0

        # Merge these counts with the cell data.
        vector_df_merged = merged_df.merge(cat_counts, on="Object ID", how="left")
        # Fill missing counts with zero for cells with no Plasmodesmata votes.
        vector_df_merged[["X", "Y", "Z"]] = vector_df_merged[["X", "Y", "Z"]].fillna(0)

        # Compute total connection votes per cell.
        vector_df_merged["total_conn"] = (
            vector_df_merged["X"] + vector_df_merged["Y"] + vector_df_merged["Z"]
        )
        # Compute fractions for each category (defining fraction as zero when total is zero)
        vector_df_merged["frac_X"] = vector_df_merged["X"] / vector_df_merged[
            "total_conn"
        ].replace({0: np.nan})
        vector_df_merged["frac_Y"] = vector_df_merged["Y"] / vector_df_merged[
            "total_conn"
        ].replace({0: np.nan})
        vector_df_merged["frac_Z"] = vector_df_merged["Z"] / vector_df_merged[
            "total_conn"
        ].replace({0: np.nan})
        vector_df_merged[["frac_X", "frac_Y", "frac_Z"]] = vector_df_merged[
            ["frac_X", "frac_Y", "frac_Z"]
        ].fillna(0)

        # Store the vector categorization info per cell.
        vector_dfs[dataset] = vector_df_merged.copy()

    # Save vector data for future use
    vector_cache_file = os.path.join(DATA_CACHE_DIR, "vector_dfs.pkl")
    with open(vector_cache_file, "wb") as f:
        pickle.dump(vector_dfs, f)
    print(f"Saved vector data to {vector_cache_file}")

    # Optional: You could also (if desired) produce individual scatter plots per dataset here similar to before.
    # For brevity, these plots are omitted from this block.
# %%
# ---------------------------
# COMBINED PLOTTING FOR VECTOR CATEGORIZATION
# ---------------------------
# For each vector category (here the fraction of connections of type "Z", "X", or "Y"),
# we produce one figure with three subplots (one per COM coordinate).
# Each subplot plots the binned average fraction (with SEM error bars) for that category.
vec_cat_cols = ["frac_Z", "frac_X", "frac_Y"]
cat_labels = {"frac_Z": "Z", "frac_X": "X", "frac_Y": "Y"}

for frac_col in vec_cat_cols:
    fig, axs = plt.subplots(1, len(com_coords), figsize=(18, 5))
    for i, coord in enumerate(com_coords):
        ax = axs[i]
        # Create common bin edges for this coordinate across all datasets.
        all_vals = np.concatenate([vector_dfs[ds][coord].values for ds in datasets])
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Plot data for each dataset.
        for ds in datasets:
            df = vector_dfs[ds].copy()
            df["bin"] = pd.cut(
                df[coord], bins=bin_edges, labels=bin_centers, include_lowest=True
            )
            # Group the data by bin and compute mean and SEM for the fraction.
            group = df.groupby("bin", observed=False)[frac_col]
            mean_vals = group.mean()
            std_vals = group.std()
            count_vals = group.count()
            sem_vals = std_vals / np.sqrt(count_vals)
            valid = count_vals > 0
            ax.errorbar(
                mean_vals.index.astype(float)[valid],
                mean_vals.values[valid],
                yerr=sem_vals.values[valid],
                marker="o",
                linestyle="-",
                capsize=5,
                label=ds,
            )
        ax.set_xlabel(coord)
        ax.set_ylabel(f"Fraction of {cat_labels[frac_col]} connections")
        ax.set_title(f"{cat_labels[frac_col]} fraction vs {coord}")
        ax.set_ylim(0, 1)
        ax.legend()
    plt.suptitle(
        f"Vector Categorization: Fraction of {cat_labels[frac_col]} connections",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        f"/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/analysis/figures/all_datasets_fraction_of_{cat_labels[frac_col]}_connections",
        dpi=300,
    )  # Save to PNG file, high resolution (300 dpi)
    plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# PARAMETERS & DATASET LIST
datasets = ["jrc_22ak351-leaf-2l", "jrc_22ak351-leaf-3m", "jrc_22ak351-leaf-3r"]
max_distance = 7500.0  # nm
nbins = 1200  # number of bins for the RDF histogram

# Define common bins from 0 to max_distance
bins = np.linspace(0, max_distance, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
dr = bins[1] - bins[0]


def compute_histogram(dists, bins, max_distance, total_valid_pairs):
    """
    Compute the histogram for the observed pairwise distances and calculate
    the radial distribution function (g(r)) using a disk (2D) normalization.

    Parameters:
        dists (ndarray): Array of pairwise distances.
        bins (ndarray): Array of bin edges.
        max_distance (float): The maximum distance used for the histogram.
        total_valid_pairs (int): The total number of valid pairs used for normalization.

    Returns:
        hist (ndarray): The observed histogram counts.
        g_r (ndarray): The radial distribution function values.
    """
    # Histogram the observed distances
    hist, _ = np.histogram(dists, bins=bins)

    # Total area of the disk with radius = max_distance
    A_total = np.pi * max_distance**2
    # Compute the area of each annulus corresponding to the histogram bins
    annulus_areas = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    # Compute the expected counts (if points were uniformly distributed)
    expected = total_valid_pairs * (annulus_areas / A_total)

    # Compute the RDF: g(r) = observed / expected
    g_r = hist / expected

    return hist, g_r


def calculate_rdf(coords, max_distance, bins):
    """
    Calculate the radial distribution function (RDF) for a set of 3D coordinates.
    This function builds a KD-tree, finds all pairs within max_distance,
    computes pairwise distances, and uses the compute_histogram function to
    obtain the RDF.

    Parameters:
        coords (ndarray): Array of 3D coordinates (shape: (N, 3)).
        max_distance (float): Maximum distance to consider for pair search.
        bins (ndarray): Array of bin edges used for the histogram.

    Returns:
        dists (ndarray): Array of computed pairwise distances.
        g_r (ndarray): The computed radial distribution function.
        total_valid_pairs (int): The total number of valid pairs found.
    """
    # Build a cKDTree for efficient neighbor search
    tree = cKDTree(coords)
    # Get unique pairs of points that are within the maximum distance
    pairs = tree.query_pairs(max_distance)
    pairs_array = np.array(list(pairs))

    # Check if we have any pairs at all
    if pairs_array.size == 0:
        return np.array([]), np.array([]), 0

    # Extract the coordinates of the paired points
    point_pairs1 = coords[pairs_array[:, 0]]
    point_pairs2 = coords[pairs_array[:, 1]]

    # Compute the difference vectors for each pair and their Euclidean distances
    diffs = point_pairs1 - point_pairs2
    dists = np.linalg.norm(diffs, axis=1)
    total_valid_pairs = len(dists)

    # Calculate the histogram and RDF using the dedicated function
    _, g_r = compute_histogram(dists, bins, max_distance, total_valid_pairs)

    return dists, g_r, total_valid_pairs


# %%

# Dictionary to store RDF results for each dataset
rdf_results = {}

if USE_EXISTING_DATA:
    # Load existing RDF data
    print("Loading existing RDF data from cache...")
    rdf_cache_file = os.path.join(DATA_CACHE_DIR, "rdf_results.pkl")
    if os.path.exists(rdf_cache_file):
        with open(rdf_cache_file, "rb") as f:
            rdf_results = pickle.load(f)
        print(f"Loaded RDF data for {len(rdf_results)} datasets from {rdf_cache_file}")
    else:
        print(
            f"WARNING: RDF cache file not found at {rdf_cache_file}. Running analysis instead."
        )
        USE_EXISTING_DATA = False

if not USE_EXISTING_DATA:
    for ds in datasets:
        # Construct file path and load the CSV file for each dataset
        pd_file = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{ds}/plasmodesmata_cleaned_lines_assigned_to_2_nearest_cells.csv"
        df = pd.read_csv(pd_file)

        # Use the full 3D center of mass (COM) coordinates (Z, Y, X)
        coords = df[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].values
        N = coords.shape[0]

        # Check that there are enough points for an RDF calculation
        if N < 2:
            print(f"Dataset {ds} has too few points ({N}) for RDF calculation.")
            continue

        print(f"Calculating RDF for dataset {ds}")
        dists, g_r, total_valid_pairs = calculate_rdf(coords, max_distance, bins)

        if total_valid_pairs == 0:
            print(f"No valid pairs found for dataset {ds}.")
            continue

        rdf_results[ds] = (bin_centers, g_r)

    # Save RDF data for future use
    rdf_cache_file = os.path.join(DATA_CACHE_DIR, "rdf_results.pkl")
    with open(rdf_cache_file, "wb") as f:
        pickle.dump(rdf_results, f)
    print(f"Saved RDF data to {rdf_cache_file}")

# Optionally, plot the RDF for each dataset

plt.figure(figsize=(8, 6))
plt.axhline(y=1, color="black")
for ds in datasets:
    if ds in rdf_results:
        r_vals, g_r = rdf_results[ds]
        plt.plot(r_vals, g_r, linestyle="-", label=ds)
plt.xlabel("r (nm)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function (3D distances, 2D disk normalization)")
plt.legend()
plt.semilogx()
# plt.xlim([10,1E4])
# plt.ylim([0.01, 18])
# plt.loglog()
plt.tight_layout()
plt.savefig(
    "/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/analysis/figures/rdf_plasmodesmata_disk_norm.png",
    dpi=300,
)
plt.show()


# %%
# rdf by region
# Dictionary to store RDF results for each dataset
rdf_results_regions = {}
datasets_regions = {
    "jrc_22ak351-leaf-2l": {
        "epidermis_start": [0, 20_000],
        "palisade": [20_000, 50_000],
        "mesophyl": [50_000, 130_000],
        # "mesophyl_0": [50_000, 71_000],
        # "vasculature": [71_000, 100_000],
        # "mesophyl_1": [100_000, 130_000],
        "epidermis_end": [130_000, np.inf],
    },
    "jrc_22ak351-leaf-3m": {
        "epidermis_start": [0, 48_000],
        "mesophyl": [48_000, 230_000],
        "epidermis_end": [230_000, np.inf],
    },
    "jrc_22ak351-leaf-3r": {
        "epidermis_start": [0, 20_000],
        "palisade": [20_000, 53_000],
        "mesophyl": [53_000, 145_000],
        "epidermis_end": [145_000, np.inf],
    },
}

if USE_EXISTING_DATA:
    # Load existing RDF regions data
    print("Loading existing RDF regions data from cache...")
    rdf_regions_cache_file = os.path.join(DATA_CACHE_DIR, "rdf_results_regions.pkl")
    if os.path.exists(rdf_regions_cache_file):
        with open(rdf_regions_cache_file, "rb") as f:
            rdf_results_regions = pickle.load(f)
        print(
            f"Loaded RDF regions data for {len(rdf_results_regions)} datasets from {rdf_regions_cache_file}"
        )
    else:
        print(
            f"WARNING: RDF regions cache file not found at {rdf_regions_cache_file}. Running analysis instead."
        )
        USE_EXISTING_DATA = False

if not USE_EXISTING_DATA:
    for ds, regions in datasets_regions.items():
        # Construct file path and load the CSV file for each dataset
        pd_file = f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{ds}/plasmodesmata_cleaned_lines_assigned_to_2_nearest_cells.csv"
        df = pd.read_csv(pd_file)
        print(f"{ds}: {len(df)} of plasmodesmata")
        rdf_results_regions[ds] = {}
        # Use the full 3D center of mass (COM) coordinates (Z, Y, X)
        coords = df[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].values
        for region, limits in regions.items():
            # Filter the coordinates based on the region
            region_coords = coords[
                (coords[:, 2] >= limits[0]) & (coords[:, 2] < limits[1])
            ]
            N = region_coords.shape[0]

            # Check that there are enough points for an RDF calculation
            if N < 2:
                print(f"Dataset {ds} has too few points ({N}) for RDF calculation.")
                continue

            print(f"Calculating RDF for dataset {ds}, region {region}")
            dists, g_r, total_valid_pairs = calculate_rdf(
                region_coords, max_distance, bins
            )
            print(
                f"Calculated RDF for dataset {ds}, region {region}, {len(region_coords)}, {len(dists)}"
            )

            if total_valid_pairs == 0:
                print(f"No valid pairs found for dataset {ds}.")
                continue

            rdf_results_regions[ds][region] = (bin_centers, g_r)

    # Save RDF regions data for future use
    rdf_regions_cache_file = os.path.join(DATA_CACHE_DIR, "rdf_results_regions.pkl")
    with open(rdf_regions_cache_file, "wb") as f:
        pickle.dump(rdf_results_regions, f)
    print(f"Saved RDF regions data to {rdf_regions_cache_file}")
# %%
plt.figure(figsize=(8, 6))
for ds, regions in rdf_results_regions.items():
    for region in regions.keys():
        plt.axhline(y=1, color="black")
        r_vals, g_r = rdf_results_regions[ds][region]
        plt.plot(r_vals, g_r, linestyle="-", label=region)
    plt.xlabel("r (nm)")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function for {ds}")
    plt.legend()
    plt.ylim([0, 5.5])
    plt.semilogx()
    # plt.xlim([10,1E4])
    # plt.ylim([0.01, 18])
    # plt.loglog()
    plt.tight_layout()
    plt.savefig(
        f"/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/analysis/figures/{ds}_rdf_regions.png",
        dpi=300,
    )
    plt.show()
# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # Define number of steps for a smooth gradient
# nsteps = 256

# # Generate a list of colors based on the formula (1.0-x, x, 1.0) for x in [0,1]
# colors = [(1.0 - i, i, 1.0) for i in np.linspace(0, 1, nsteps)]

# # Create a custom colormap
# custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# # Create a dummy gradient from 0 to 1 and then tile it vertically
# gradient = np.linspace(0, 1, nsteps)
# gradient = np.tile(gradient, (50, 1))

# # Plot the gradient image.
# # The extent argument re-scales the x-axis: it maps the original data [0, 1] to [0, 20]
# plt.figure(figsize=(8, 2))
# plt.imshow(gradient, aspect="auto", cmap=custom_cmap, extent=(0, 20, 0, 50))

# # Generate the colorbar with a horizontal orientation
# plt.colorbar(orientation="horizontal")
# plt.title("Custom Color Bar: (1.0-x, x, 1.0)")
# plt.xlabel("Value (scaled from 0 to 20)")
# plt.yticks([])
# plt.show()
# # %%

# %%
#  mesh distances

import potpourri3d as pp3d
import trimesh

mesh = trimesh.load_mesh(
    "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3m/cell/meshes/20.ply"
)

# Note: This section reuses rdf_results_regions from the cached data above if USE_EXISTING_DATA is True
# If you need to recompute, set USE_EXISTING_DATA = False at the top of the file
# solver = pp3d.PointCloudHeatSolver(P)


# %%
import pandas as pd
import potpourri3d as pp3d
import trimesh
import numpy as np
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt

# Read the plasmodesmata CSV and convert list-like string columns to actual lists
nbins = 6000  # number of bins for the RDF histogram

# Define common bins from 0 to max_distance
bins = np.linspace(0, 50_000, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])


def compute_histogram(dists, bins, surface_area):
    # Histogram the observed distances
    hist, _ = np.histogram(dists, bins=bins)
    # Compute the area of each annulus corresponding to the histogram bins
    annulus_areas = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    # Compute the expected counts (if points were uniformly distributed)
    expected = len(dists) * (annulus_areas / surface_area)

    # Compute the RDF: g(r) = observed / expected
    g_r = hist / expected

    return hist, g_r


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

closest, dists, _ = trimesh.proximity.closest_point(
    cell_mesh, cell_plasmodesmata_coords
)
P = np.vstack([cell_mesh.vertices, closest])
solver = pp3d.PointCloudHeatSolver(P)
all_dists = np.zeros((num_plasmodesmata, num_plasmodesmata))
for idx in tqdm(range(num_vertices, num_vertices + num_plasmodesmata)):
    # Compute the distance from the point to the mesh
    dists = solver.compute_distance(idx)
    all_dists[idx - num_vertices] = dists[num_vertices:]

import pandas as pd
import potpourri3d as pp3d
import trimesh
import numpy as np
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt

# Read the plasmodesmata CSV and convert list-like string columns to actual lists
nbins = 6000  # number of bins for the RDF histogram

# Define common bins from 0 to max_distance
bins = np.linspace(0, 50_000, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])


def compute_histogram(dists, bins, surface_area):
    # Histogram the observed distances
    hist, _ = np.histogram(dists, bins=bins)
    # Compute the area of each annulus corresponding to the histogram bins
    annulus_areas = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    # Compute the expected counts (if points were uniformly distributed)
    expected = len(dists) * (annulus_areas / surface_area)

    # Compute the RDF: g(r) = observed / expected
    g_r = hist / expected

    return hist, g_r


# %%
import pygeodesic.geodesic as geodesic
from remesh import insert_points_into_mesh_original
import pandas as pd
import numpy as np
import ast
import trimesh
from tqdm import tqdm

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
print("Inserting points into mesh...")
updated_vertices, updated_faces = insert_points_into_mesh_original(
    cell_mesh, cell_plasmodesmata_coords
)
all_dists = []
print("got to here")
geoalg = geodesic.PyGeodesicAlgorithmExact(updated_vertices, updated_faces)
for idx in tqdm(range(num_vertices, num_vertices + num_plasmodesmata)):
    if idx + 1 == len(updated_vertices):
        break
    distances, _ = geoalg.geodesicDistances(
        [idx], list(range(idx + 1, len(updated_vertices)))
    )
    # Compute the distance from the point to the mesh
    all_dists.append(distances)


# %%
x, g_r = compute_histogram(
    all_dists[np.triu_indices_from(all_dists, k=1)], bins, cell_mesh.area
)
plt.figure(figsize=(8, 6))
# for ds, regions in rdf_results_regions.items():
#     for region in regions.keys():
plt.axhline(y=1, color="black")
#         r_vals, g_r = rdf_results_regions[ds][region]
plt.plot(bin_centers, g_r, linestyle="-")  # , label=region)
plt.xlabel("r (nm)")
plt.ylabel("g(r)")
plt.title(f"Radial Distribution Function for {dataset}")
plt.legend()
# plt.semilogx()
# plt.xlim([10,1E4])
# plt.ylim([0.01, 18])
plt.loglog()
plt.tight_layout()
plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    closest[:, 0], closest[:, 1], closest[:, 2], c=dists[num_vertices:], cmap="viridis"
)
cbar = plt.colorbar(scatter)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=dists, cmap="viridis")
cbar = plt.colorbar(scatter)
