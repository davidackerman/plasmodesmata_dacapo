# %%
import os
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from scipy.ndimage import binary_dilation, distance_transform_edt
import numpy as np

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from scipy.ndimage import binary_dilation, distance_transform_edt
import numpy as np
import pandas as pd
from image_data_interface import ImageDataInterface
from scipy.ndimage import distance_transform_edt
import fastmorph

dataset = "jrc_22ak351-leaf-3mb"  # 3r, 3m
cell_segmentation_paths = pd.read_csv("cell_segmentation_paths.csv")
cell_segmentation_path = cell_segmentation_paths[
    cell_segmentation_paths["dataset"] == dataset
].iloc[0]["path"]

if dataset.endswith("b"):
    # the data is at 32 nm, but wasnt saved with the resolution, so will do it ourselves
    output_voxel_size = Coordinate([32, 32, 32])
    idi = ImageDataInterface(cell_segmentation_path)
    total_roi = idi.roi * output_voxel_size
    cells = idi.to_ndarray_ts()
    cells = fastmorph.erode(cells)

else:
    output_voxel_size = Coordinate([256, 256, 256])
    idi = ImageDataInterface(
        cell_segmentation_path, output_voxel_size=output_voxel_size
    )
    total_roi = idi.roi
    cells = idi.to_ndarray_ts()
inclusive_mask = 1 - (cells > 0)
distance_from_cell = distance_transform_edt(cells == 0)

for d in range(1, 10):
    inclusive_mask_dilated = binary_dilation(inclusive_mask, iterations=d)
    output_ds = prepare_ds(
        "/nrs/cellmap/ackermand/cellmap/leaf-gall/prediction_masks.zarr",
        f"dilation_iterations_{d}_{dataset}/s0",
        total_roi=total_roi,
        voxel_size=output_voxel_size,
        dtype=np.uint8,
        write_size=Coordinate(np.array([64, 64, 64]) * output_voxel_size[0]),
        delete=True,
    )
    output_ds[total_roi] = (inclusive_mask_dilated) & (distance_from_cell <= d)


# %%
import fastmorph
import numpy as np

a = np.array([[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]]).astype(
    np.uint8
)
fastmorph.erode(a)
# %%
