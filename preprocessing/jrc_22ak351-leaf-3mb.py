# %%

# Write out annotations
import getpass

username = getpass.getuser()
organelle = "plasmodesmata"
dataset = "jrc_22ak351-leaf-3mb"
# %%
import annotation_processing_utils.process.cylindrical_annotations
from importlib import reload

reload(annotation_processing_utils.process.cylindrical_annotations)
from annotation_processing_utils.process.cylindrical_annotations import (
    CylindricalAnnotations,
)

radius = 4
ca = CylindricalAnnotations(
    organelle=organelle,
    training_validation_test_roi_info_yaml=f"/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/{dataset}/training_validation_test_roi_info.yaml",
    output_mask_zarr=f"/nrs/cellmap/{username}/cellmap/{organelle}/annotation_intersection_masks.zarr",
    output_gt_zarr=f"/nrs/cellmap/{username}/cellmap/{organelle}/annotations_as_cylinders.zarr",
    output_training_points_zarr=f"/nrs/cellmap/{username}/cellmap/{organelle}/training_points.zarr",
    output_annotations_directory=f"/nrs/cellmap/{username}/cellmap/{organelle}/neuroglancer_annotations",
    raw_path="/nrs/cellmap/data/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb.n5/em/fibsem-uint8/s1",
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

# import time

# # lazy_results = []
# for lsds_to_affs_weight_ratio in [0.5, 1.0, 2.0]:
#     for batch_size in [2, 8]:
#         # print current time
#         print(time.ctime())
#         ca.create_dacapo_run(
#             repetitions=2,
#             lsds_to_affs_weight_ratio=lsds_to_affs_weight_ratio,
#             batch_size=batch_size,
#             validation_interval=1_000_000,
#         )


# cluster = LocalCluster(n_workers=10, threads_per_worker=1, host="0.0.0.0")
# with Client(cluster) as client:
#     dask.compute(*lazy_results)

# %%
# Postprocessing
import annotation_processing_utils.postprocess.get_best
from importlib import reload

reload(annotation_processing_utils.postprocess.get_best)
from annotation_processing_utils.postprocess.get_best import GetBest

print(dataset)
gb = GetBest(
    f"/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/processing_yamls/{dataset}.yaml"
)
failed = gb.f1_score()
gb.plot_f1_scores("validation", plot_type="histogram", merge_repetitions=True)

print(failed)
# %%
gb.all_f1_scores_df
# %%
# from annotation_processing_utils.process.training_validation_test_roi_calculator import (
#     TrainingValidationTestRoiCalculator,
# )

# calc = TrainingValidationTestRoiCalculator(
#     "/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/jrc_22ak351-leaf-3mb/training_validation_test_roi_info.yaml"
# )
# calc.get_training_validation_test_rois()
# calc.rois_dict
# %%
