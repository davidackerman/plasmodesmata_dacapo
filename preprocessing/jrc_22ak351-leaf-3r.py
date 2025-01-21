# %%

# Write out annotations
from annotation_processing_utils.process.cylindrical_annotations import (
    CylindricalAnnotations,
)
import getpass

username = getpass.getuser()
organelle = "plasmodesmata"
dataset = "jrc_22ak351-leaf-3r"

# %%
radius = 4
ca = CylindricalAnnotations(
    organelle=organelle,
    training_validation_test_roi_info_yaml=f"/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/{dataset}/training_validation_test_roi_info.yaml",
    output_mask_zarr=f"/nrs/cellmap/{username}/cellmap/{organelle}/annotation_intersection_masks.zarr",
    output_gt_zarr=f"/nrs/cellmap/{username}/cellmap/{organelle}/annotations_as_cylinders.zarr",
    output_training_points_zarr=f"/nrs/cellmap/{username}/cellmap/{organelle}/training_points.zarr",
    output_annotations_directory=f"/nrs/cellmap/{username}/cellmap/{organelle}/neuroglancer_annotations",
    dataset=dataset,
    radius=radius,
)
ca.standard_processing()
# save ca to pkl
ca.save(f"./{dataset}_cylindrical_annotations.pkl")


# %%
# Save out dacapo runs
# import dask
# from dask.distributed import LocalCluster, Client

import time

# lazy_results = []
for lsds_to_affs_weight_ratio in [0.5]:
    for batch_size in [2]:
        # print current time
        print(time.ctime())
        ca.create_dacapo_run(
            repetitions=2,
            lsds_to_affs_weight_ratio=lsds_to_affs_weight_ratio,
            batch_size=batch_size,
            validation_interval=1_000_000,
        )

# # %%
# # Visualize pipeline
# from dacapo.store.create_store import create_config_store
# from dacapo.experiments import Run

# config_store = create_config_store()
# run_config = config_store.retrieve_run_config(
#     "finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00015_bs_6__0"
# )
# run = Run(run_config)
# run.visualize_pipeline()
# # %% create prediction mask

# from funlib.persistence import prepare_ds
# from funlib.geometry import Roi, Coordinate
# from scipy.ndimage import binary_dilation
# import numpy as np

# from funlib.persistence import open_ds, prepare_ds
# from funlib.geometry import Roi, Coordinate
# from scipy.ndimage import binary_dilation
# import numpy as np
# import pandas as pd
# from image_data_interface import ImageDataInterface

# cell_segmentation_paths = pd.read_csv("cell_segmentation_paths.csv")
# cell_segmentation_path = cell_segmentation_paths[
#     cell_segmentation_paths["dataset"] == dataset
# ].iloc[0]["path"]

# output_voxel_size = Coordinate([256, 256, 256])
# idi = ImageDataInterface(cell_segmentation_path, output_voxel_size=output_voxel_size)

# inclusive_mask = 1 - (idi.to_ndarray_ts() > 0)

# for iterations in range(1, 4):
#     inclusive_mask_dilated = binary_dilation(inclusive_mask, iterations=iterations)

#     output_ds = prepare_ds(
#         "/nrs/cellmap/ackermand/cellmap/leaf-gall/prediction_masks.zarr",
#         f"dilation_iterations_{iterations}_{dataset}/s0",
#         total_roi=idi.roi,
#         voxel_size=output_voxel_size,
#         dtype=np.uint8,
#         write_size=Coordinate(np.array([64, 64, 64]) * output_voxel_size[0]),
#         delete=True,
#     )
#     output_ds[idi.roi] = inclusive_mask_dilated
