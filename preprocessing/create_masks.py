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
import cc3d 
import edt
for dataset in ["jrc_22ak351-leaf-2lb","jrc_22ak351-leaf-3mb","jrc_22ak351-leaf-3rb"]:
    print(f"Processing {dataset}...")
    cell_segmentation_paths = pd.read_csv("cell_segmentation_paths.csv")
    cell_segmentation_path = cell_segmentation_paths[
        cell_segmentation_paths["dataset"] == dataset
    ].iloc[0]["path"]

    if dataset.endswith("b"):
        # the data is at 32 nm, but wasnt saved with the resolution, so will do it ourselves (except for 3mb)
        output_voxel_size = Coordinate([32, 32, 32])
        raw_scale = "s3"
        idi = ImageDataInterface(cell_segmentation_path)
        if dataset == "jrc_22ak351-leaf-3mb":
            total_roi = idi.roi
        else:
            total_roi = idi.roi * output_voxel_size
        cells = idi.to_ndarray_ts()
    else:
        output_voxel_size = Coordinate([256, 256, 256])
        idi = ImageDataInterface(
            cell_segmentation_path, output_voxel_size=output_voxel_size
        )
        total_roi = idi.roi
        cells = idi.to_ndarray_ts()
    
    # mask out 0 raw
    raw = ImageDataInterface(f"/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8/{raw_scale}").to_ndarray_ts()
    if raw.shape != cells.shape:
        padded_raw = np.zeros(cells.shape, dtype=raw.dtype)
        padded_raw[
            : raw.shape[0], : raw.shape[1], : raw.shape[2]
        ] = raw
        raw = padded_raw
    raw_background_cc = cc3d.connected_components(raw==0, connectivity=26, binary_image=True)
    raw_ids, raw_counts = np.unique(raw_background_cc, return_counts=True)
    raw_counts = raw_counts[raw_ids != 0]
    raw_ids = raw_ids[raw_ids != 0]
    # get id with most counts
    largest_raw_id = raw_ids[np.argmax(raw_counts)]
    raw_background = raw_background_cc==largest_raw_id
    raw_foreground = 1 - raw_background
 
    max_iterations = 10

    # expand raw_mask by 1
    raw_background_dilated = fastmorph.dilate(raw_background, iterations=1)
    cells_signed_distance_transform = edt.sdf(cells)
    raw_background_dilated_distance_transform = edt.edt(1-raw_background_dilated)
    cells_inside = np.abs(cells_signed_distance_transform)
    cells_inside[cells_signed_distance_transform <0] = -1

    # invalid if distance to raw_background_dilated is less than distance inside cell, means its bordering black 
    invalid_voxels = raw_background_dilated_distance_transform<=cells_inside

    for d in range(1, max_iterations):
        print(f"  Creating mask for dilation {d}...")
        output_ds = prepare_ds(
            "/nrs/cellmap/ackermand/cellmap/leaf-gall/prediction_masks.zarr",
            f"dilation_iterations_{d}_{dataset}/s0",
            total_roi=total_roi,
            voxel_size=output_voxel_size,
            dtype=np.uint8,
            write_size=Coordinate(np.array([64, 64, 64]) * output_voxel_size[0]),
            delete=True,
        )
        output_ds[total_roi] = (np.abs(cells_signed_distance_transform) <=d) & (raw_foreground) & (~invalid_voxels)

  # %%
# import matplotlib.pyplot as plt
# import fastremap
# edt.edt(cells_signed_distance_transform==0)
# plt.figure()
# # cells_binarized = cells>0
# # cells_binarized[raw_background_dilated] = 1
# # cells_binarized_dt = edt.edt(cells_binarized)
# # cell_ids = fastremap.unique(cells)
# # cell_dt = np.zeros_like(cells,dtype=np.float32)
# # for cell_id in cell_ids:
# #     print(f"Processing cell {cell_id}...")
# #     if cell_id == 0:
# #         continue
# #     cell_mask = cells==cell_id
# #     cell_mask[raw_background_dilated] = 1
# #     cell_mask_dt = edt.edt(cell_mask)
# #     cell_mask_dt[cell_mask==0] = 0
# #     cell_dt += cell_mask_dt
# plt.imshow(cell_mask_dt[100])
# plt.figure()
# plt.imshow((cells_signed_distance_transform[100]<9))
# plt.figure()
# plt.imshow(cells[100])
# plt.figure()
# # cells_inside = cells_signed_distance_transform
# # cells_inside[cells_inside <0] = np.inf
# # cells_inside = np.abs(cells_inside)
# # cells_outside = np.abs(cells_signed_distance_transform) * (cells_signed_distance_transform < 0)
# plt.imshow(cells_inside[100])
# #plt.figure()
# #plt.imshow(cells_outside[100])
# invalid_voxels = raw_background_dilated_distance_transform<=cells_inside
# plt.figure()
# o = (np.abs(cells_signed_distance_transform) <=9) & (raw_foreground) & (~invalid_voxels)
# plt.imshow(o[100])
# plt.figure()
# plt.imshow(invalid_voxels[100])

#%% get cell instances
from image_data_interface import ImageDataInterface
from postprocessing.zarr_util import create_multiscale_dataset
import pandas as pd
import cc3d
from funlib.geometry import Coordinate
import numpy as np

voxel_size_dict = {
    "jrc_22ak351-leaf-3m": 512,
    "jrc_22ak351-leaf-2l": 256,
    "jrc_22ak351-leaf-3r": 128,
}
cell_segmentation_paths = pd.read_csv("cell_segmentation_paths.csv")
for dataset in ["jrc_22ak351-leaf-3m", "jrc_22ak351-leaf-3r", "jrc_22ak351-leaf-2l"]:
    cell_value = 1 if dataset.endswith("3m") else 60
    cell_segmentation_path = cell_segmentation_paths[
        cell_segmentation_paths["dataset"] == dataset
    ].iloc[0]["path"]
    idi = ImageDataInterface(cell_segmentation_path)
    cells = idi.to_ndarray_ts()
    connected_components = cc3d.connected_components(
        cells == cell_value, connectivity=6, binary_image=True
    )
    connected_components = connected_components.astype(
        np.min_scalar_type(connected_components.max())
    )
    # for d in range(1, 10):
    #     inclusive_mask_dilated = binary_dilation(inclusive_mask, iterations=d)
    voxel_size = voxel_size_dict[dataset]
    roi = idi.roi * voxel_size / idi.voxel_size
    output_ds = create_multiscale_dataset(
        output_path=f"/nrs/cellmap/ackermand/cellmap/leaf-gall/{dataset}.zarr/cell",
        dtype=connected_components.dtype,
        voxel_size=3 * [voxel_size],
        total_roi=idi.roi * voxel_size / idi.voxel_size,
        write_size=Coordinate(np.array([64, 64, 64]) * voxel_size),
    )
    output_ds[roi] = connected_components

# %%
# import fastmorph
# import numpy as np

# a = np.array([[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]]).astype(
#     np.uint8
# )
# fastmorph.erode(a)
# # %%
# # write out as zarrs to use for cellpose
# from tifffile import imwrite
# from preprocessing.image_data_interface import ImageDataInterface
# import numpy as np


# def zarr_to_tif(
#     zarr_paths,
#     tif_path,
#     skip=1,
#     invert=False,
#     roi=None,
# ):
#     for data_type, zarr_path in zarr_paths.items():
#         # Dictionary defining each orientation with the appropriate slicing lambda
#         orientations = {
#             "YX": {"axis": 0, "extract": lambda img, i: img[i, :, :]},  # slices along Z
#             "XZ": {"axis": 1, "extract": lambda img, i: img[:, i, :]},  # slices along Y
#             "YZ": {"axis": 2, "extract": lambda img, i: img[:, :, i]},  # slices along X
#         }

#         print(f"Loading {zarr_path}...")
#         idi = ImageDataInterface(zarr_path)
#         if roi is not None:
#             img = idi.to_ndarray_ts(roi)
#         else:
#             img = idi.to_ndarray_ts()

#         if invert:
#             print("Inverting ...")
#             if dataset.dtype == np.uint8:
#                 img = 255 - img
#             else:  # assume float
#                 img = 1 - img

#         # Assuming img is your 3D numpy array with shape (Z, Y, X)
#         # img = np.load('your_3d_image.npy')

#         # Loop over each orientation and save slices
#         all_output_dir = tif_path + f"/all/"
#         os.makedirs(all_output_dir, exist_ok=True)
#         for orient, params in orientations.items():
#             empty_masks = []

#             # Determine the number of slices in the relevant axis
#             num_slices = img.shape[params["axis"]]
#             output_dir = tif_path + f"/{orient}/"
#             os.makedirs(output_dir, exist_ok=True)
#             suffix = "_masks.tif" if data_type == "masks" else ".tif"
#             for i in range(0, num_slices, skip):
#                 slice_img = params["extract"](img, i)
#                 filename = output_dir + f"frame_{(i//skip):03d}{suffix}"

#                 if data_type == "masks" and np.sum(slice_img) == 0:
#                     empty_masks.append(f"frame_{(i//skip):03d}")

#                 imwrite(filename, slice_img)
#                 print(f"Saved {filename}")
#                 imwrite(
#                     all_output_dir + f"{orient}_frame_{(i//skip):03d}{suffix}",
#                     slice_img,
#                 )
#             # remove empty
#             for empty_mask in empty_masks:
#                 for suffix in [".tif", "_masks.tif"]:
#                     os.system(f"rm {output_dir}{empty_mask}{suffix}")
#                     os.system(f"rm {all_output_dir}{orient}_{empty_mask}{suffix}")


# # zarr_to_tif(
# #     f"/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8,
# #     tif_path=f"/nrs/cellmap/ackermand/cellmap/leaf-gall/cellpose/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.tif",
# #     scale=5,
# # )

# zarr_to_tif(
#     {
#         "raw": "/nrs/cellmap/data/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb.n5/em/fibsem-uint8/s3",
#         "masks": "/groups/cellmap/cellmap/parkg/for Aubrey/3mb_s3.zarr/jrc_22ak351-leaf-3mb_cells",
#     },
#     tif_path="/nrs/cellmap/ackermand/cellmap/leaf-gall/cellpose/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb",
#     skip=50,
# )


# # %%
# from tifffile.tifffile import imread

# im1 = imread(
#     "/nrs/cellmap/ackermand/cellmap/leaf-gall/cellpose/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb/XZ/frame_005_masks.tif"
# )
# import matplotlib.pyplot as plt

# plt.imshow(im1)

# # %%
