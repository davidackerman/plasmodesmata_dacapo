import os
from datetime import datetime
from pathlib import Path
import yaml


def do_submission(logdir, run, iteration, roi_name, validation_or_test, dataset):
    os.system(
        f"bsub -P cellmap -n 64 -o {logdir}/rusty_{iteration}_{roi_name}_{validation_or_test}.o -e {logdir}/rusty_{iteration}_{roi_name}_{validation_or_test}.e python validation_rusty_mws.py {run} {iteration} {roi_name} {validation_or_test} {dataset}"
    )


yaml_path = "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/validation_and_test/yamls/combined_yamls/jrc_22ak351-leaf-3r_2023-12-15.yml"
with open(yaml_path, "r") as stream:
    yml = yaml.safe_load(stream)

dataset = yml["dataset"]

iterations_start, iterations_end, iterations_step = yml["iterations"]


for run in yml["runs"]:
    dirname = datetime.now().strftime(f"%Y%m%d/{run}/%H%M%S")
    logdir = Path(
        f"/nrs/cellmap/ackermand/logs/plasmodesmata_dacapo/validation_and_test/{dirname}"
    )
    os.makedirs(logdir, exist_ok=True)
    for iteration in range(iterations_start, iterations_end + 1, iterations_step):
        for roi in yml["rois"]:
            do_submission(logdir, run, iteration, roi["name"], "validation", dataset)
            if roi["split_dimension"] is not None:
                do_submission(logdir, run, iteration, roi["name"], "test", dataset)

# for run, iteration, roi, _ in [
#     (
#         "finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_more_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5__1",
#         150000,
#         6,
#         "validation",
#     ),
#     (
#         "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_more_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5__0",
#         155000,
#         11,
#         "validation",
#     ),
# ]:
#     dirname = datetime.now().strftime(f"%Y%m%d/{run}/%H%M%S")
#     logdir = Path(f"logs/{dirname}")
#     os.makedirs(logdir, exist_ok=True)
#     do_submission(logdir, run, iteration, roi, "validation")

# for run, iteration, roi, validation_or_test in [
#     (
#         "finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_more_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5__1",
#         130000,
#         6,
#         "test",
#     ),
#     (
#         "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_more_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5__2",
#         105000,
#         2,
#         "validation",
#     ),
# ]:
#     dirname = datetime.now().strftime(f"%Y%m%d/{run}/%H%M%S")
#     logdir = Path(f"logs/{dirname}")
#     os.makedirs(logdir, exist_ok=True)
#     do_submission(logdir, run, iteration, roi, validation_or_test)
