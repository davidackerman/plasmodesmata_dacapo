# %%
from importlib.resources import path
import cc3d
import tifffile
import fastmorph
import fastremap
from importlib import reload
from funlib.persistence import open_ds
import pandas as pd
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi


import numpy as np
from funlib.geometry import Roi

# im = tifffile.imread(
#     "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-2l/crop350//crop350_relabel.tif"
# )
df = pd.read_csv(
    "/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/cell_segmentation_paths.csv"
)

cts_all = {}
for stage, dataset, cell_path in zip(
    df["stage"],
    df["dataset"],
    df["path"],
):
    if dataset not in [
        "jrc_22ak351-leaf-2l",
        "jrc_22ak351-leaf-3m",
        "jrc_22ak351-leaf-3r",
    ]:
        continue
    if cell_path.endswith(".tif") or cell_path.endswith(".tiff"):
        im = tifffile.imread(cell_path)
    else:
        im = ImageDataInterface(cell_path).to_ndarray_ts()
    print(f"Processing dataset {dataset}")
    print(f"Original unique labels: {len(np.unique(im[im>0]))}")
    cc = cc3d.connected_components(im, connectivity=26)
    print(f"Connected components unique labels: {len(np.unique(cc[cc>0]))}")
    # uniq, cts = fastremap.unique(
    #     cc, return_counts=True
    # )  # may be much faster than np.unique
    # cts_all[dataset] = np.sort(cts[uniq > 0])  # exclude background
    # # remove small components
    # min_size = 1000
    # to_remove = uniq[cts < min_size]
    # labels = fastremap.mask(cc, to_remove)  # set all occurances of 1,5,13 to 0
    # print(f"Removing {len(to_remove)} small components")
    filled, removed = fastmorph.fill_holes(
        cc, remove_enclosed=True, return_removed=True
    )
    filled_renumbered, _ = fastremap.renumber(filled, in_place=True)
    print(
        f"Filled unique labels: {len(np.unique(filled_renumbered[filled_renumbered>0]))}"
    )
    print(f"Removed {len(removed)} enclosed components")
    voxel_size = [128, 128, 128]
    output_idi = create_multiscale_dataset_idi(
        output_path=f"/nrs/cellmap/ackermand/cellmap/leaf-gall/{dataset}.zarr/cell_fixed",
        dtype=filled_renumbered.dtype,
        voxel_size=voxel_size,
        total_roi=Roi(
            (0, 0, 0), np.array(filled_renumbered.shape) * np.array(voxel_size)
        ),
        write_size=np.array([128, 128, 128]) * np.array(voxel_size),
    )
    output_idi.ds.data[:] = filled_renumbered
# im_relabeled = cc3d.connected_components(im, connectivity=26)
# tifffile.imwrite(
#     "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-2l/crop350//crop350_relabel_cc3d.tif",
#     im_relabeled.astype(np.uint16),
# )
# print(len(np.unique(im[im > 0])), len(np.unique(im_relabeled[im_relabeled > 0])))

# %%
