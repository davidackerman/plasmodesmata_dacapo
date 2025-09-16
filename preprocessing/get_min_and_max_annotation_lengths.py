# %%
# load pkl file
import pickle
import numpy as np

min_length_voxels = np.inf
max_length_voxels = -np.inf
radius_voxels = 4
for suffix in ["2lb", "3mb", "3rb", "2l", "3m", "3r"]:
    dataset = f"jrc_22ak351-leaf-{suffix}"
    with open(f"{dataset}_cylindrical_annotations.pkl", "rb") as file:
        data = pickle.load(file)
        ends = data.annotation_ends
        starts = data.annotation_starts
        annotation_lengths = np.linalg.norm(ends - starts, axis=1)
        annotation_lengths = annotation_lengths[annotation_lengths > 0]
        min_length_voxels = min(min_length_voxels, annotation_lengths.min())
        max_length_voxels = max(max_length_voxels, annotation_lengths.max())
        print(f"Max annotation length for {dataset}: {annotation_lengths.max()}")
        print(f"Min annotation length for {dataset}: {annotation_lengths.min()}")

# %%
voxel_volume = 8**3
min_volume_voxels = (np.floor(min_length_voxels) * np.pi * (radius_voxels) ** 2)/2
max_volume_voxels = (np.ceil(max_length_voxels) * np.pi * (radius_voxels) ** 2)*2
print(
    "Min volume (nm^3) filter for connected component cleanup: ",
    np.floor(min_volume_voxels * voxel_volume),
)
print(
    "Max volume (nm^3) filter for connected component cleanup: ",
    np.ceil(max_volume_voxels * voxel_volume),
)
# %%
import pandas as pd

df = pd.read_csv(
    "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-2l/plasmodesmata.csv"
)
# sort by Volume (nm^3)
df.sort_values(by="Volume (nm^3)", inplace=True)
df.hist(column="Volume (nm^3)", bins=100)
# %%
df
# %%