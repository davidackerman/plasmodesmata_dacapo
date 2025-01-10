# %%

# Write out annotations
from annotation_processing_utils.process.cylindrical_annotations import (
    CylindricalAnnotations,
)
import getpass

username = getpass.getuser()
organelle = "plasmodesmata"
dataset = "jrc_22ak351-leaf-3m"
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
# # %%
# import pickle

# # load pickle
# with open(f"./cylindrical_annotations.pkl", "rb") as f:
#     ca = pickle.load(f)
# print(ca.training_points)


# %%
# Save out dacapo runs
# import dask
# from dask.distributed import LocalCluster, Client


# @dask.delayed
# def lazy_create_dacapo_run(
#     lsds_to_affs_weight_ratio, batch_size, repetitions=2, validation_interval=1_000_000
# ):
#     ca.create_dacapo_run(
#         repetitions=repetitions,
#         lsds_to_affs_weight_ratio=lsds_to_affs_weight_ratio,
#         batch_size=batch_size,
#         validation_interval=1_000_000,
#     )

import time

# lazy_results = []
for lsds_to_affs_weight_ratio in [0.5, 1.0, 2.0]:
    for batch_size in [2, 8]:
        # print current time
        print(time.ctime())
        ca.create_dacapo_run(
            repetitions=2,
            lsds_to_affs_weight_ratio=lsds_to_affs_weight_ratio,
            batch_size=batch_size,
            validation_interval=1_000_000,
        )


# cluster = LocalCluster(n_workers=10, threads_per_worker=1, host="0.0.0.0")
# with Client(cluster) as client:
#     dask.compute(*lazy_results)
# %%
# Visualize pipeline
from dacapo.store.create_store import create_config_store
from dacapo.experiments import Run

config_store = create_config_store()
run_config = config_store.retrieve_run_config(
    "finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00015_bs_6__0"
)
run = Run(run_config)
run.visualize_pipeline()
# %%
