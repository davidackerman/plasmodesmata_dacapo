import pandas as pd
import potpourri3d as pp3d
import trimesh
import numpy as np
import ast

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
cell_id = 1
cell_plasmodesmata_coords = merged_df[merged_df["Cell ID"] == cell_id][
    ["Plasmodesmata COM Z (nm)", "Plasmodesmata COM Y (nm)", "Plasmodesmata COM X (nm)"]
].to_numpy()

# read in mesh
cell_mesh_file = f"/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/jrc_22ak351-leaf-3m/cell/meshes/{cell_id}.ply"
cell_mesh = trimesh.load_mesh(cell_mesh_file)
cell_mesh.vertices = cell_mesh.vertices[:, ::-1]  # vertices are in x,y,z
closest, dists, _ = trimesh.proximity.closest_point(
    cell_mesh, cell_plasmodesmata_coords
)
P = np.vstack([cell_mesh.vertices, closest])
print(len(P), P.shape)
solver = pp3d.PointCloudHeatSolver(P)
solver.compute_distance(len(P) - 1)
# %%
import yaml

# read in the yaml
with open(
    "/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/jrc_22ak351-leaf-3mb/training_validation_test_roi_info.yaml",
    "r",
) as f:
    info = yaml.load(f, Loader=yaml.FullLoader)

print(info)
# %%
import yaml

# Load YAML
with open(
    "/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/jrc_22ak351-leaf-3mb/training_validation_test_roi_info.yaml",
    "r",
) as f:
    info = yaml.safe_load(f)


# Function to scale coordinate ranges
def scale_range(range_str, factor=1):
    start, end = map(int, range_str.split("-"))
    # make sure smallest is start
    if start > end:
        start, end = end, start
    return f"{start*factor}-{end*factor}"


# Modify info in-place
for roi in info["rois_to_split"]["validation_test"]:
    roi["x"] = scale_range(roi["x"])
    roi["y"] = scale_range(roi["y"])
    roi["z"] = scale_range(roi["z"])

# Save back to YAML
out_path = "/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/jrc_22ak351-leaf-3mb/training_validation_test_roi_info_scaled.yaml"
with open(out_path, "w") as f:
    yaml.dump(info, f, sort_keys=False)
# %%
