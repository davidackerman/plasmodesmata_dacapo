import os
from datetime import datetime
from pathlib import Path
import yaml


yaml_name = "jrc_22ak351-leaf-2l_2023-12-15"
with open(
    f"/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/validation_and_test/yamls/combined_yamls/{yaml_name}.yml",
    "r",
) as stream:
    yml = yaml.safe_load(stream)


dataset = yml["dataset"]
gt_array_filename = yml["gt_array"]["filename"]
gt_array_ds_name = yml["gt_array"]["ds_name"]
mask_array_filename = yml["mask_array"]["filename"]
mask_array_ds_name = yml["mask_array"]["ds_name"]

iterations_start, iterations_end, iterations_step = yml["iterations"]
num_workers = 10
for run in yml["runs"]:
    log_dir = datetime.now().strftime(
        f"/nrs/cellmap/ackermand/logs/plasmodesmata_dacapo/validation_and_test/%Y%m%d/{run}/%H%M%S"
    )
    os.makedirs(log_dir, exist_ok=True)
    for iteration in range(iterations_start, iterations_end + 1, iterations_step):
        for roi in yml["rois"]:
            roi_name = roi["name"]
            for postprocessing_suffix in yml["postprocessing_suffixes"]:
                types = ["validation"]
                if roi["split_dimension"] is not None:
                    types.append("test")

                for validation_or_test in types:
                    test_array = yml["test_array"]
                    test_array_filename = f'{test_array["base_filename"]}/processed/{validation_or_test}/{run}/{roi_name}.n5'
                    test_array_ds_name = f"iteration_{iteration}{postprocessing_suffix}"
                    output_directory = f"/nrs/cellmap/ackermand/validation_and_testing_scores/{yaml_name}/{validation_or_test}/{run}/{roi_name}/iteration_{iteration}{postprocessing_suffix}"
                    logfile_prefix = f"{log_dir}/metrics_{iteration}_{roi_name}_{validation_or_test}_{postprocessing_suffix}"
                    args = f"{dataset} {gt_array_filename} {gt_array_ds_name} {test_array_filename} {test_array_ds_name} {mask_array_filename} {mask_array_ds_name} {output_directory} {num_workers}"
                    os.system(
                        f"bsub -P cellmap -n {num_workers} -o {logfile_prefix}.o -e {logfile_prefix}.e python instance_segmentation_metrics.py {args}"
                    )


# redos
# for run, iteration, roi_name, validation_or_test in [
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
#     log_dir = datetime.now().strftime(f"logs/%Y%m%d/{run}/%H%M%S")
#     os.makedirs(log_dir, exist_ok=True)
#     for postprocessing_suffix in yml["postprocessing_suffixes"]:
#         test_array = yml["test_array"]
#         test_array_filename = f'{test_array["base_filename"]}/processed/{validation_or_test}/{run}/{roi_name}.n5'
#         test_array_ds_name = f"iteration_{iteration}{postprocessing_suffix}"
#         output_directory = f"/nrs/cellmap/ackermand/validation_and_testing_scores/{yaml_name}/{validation_or_test}/{run}/{roi_name}/iteration_{iteration}{postprocessing_suffix}"
#         logfile_prefix = f"{log_dir}/metrics_{iteration}_{roi_name}_{validation_or_test}_{postprocessing_suffix}"
#         args = f"{dataset} {gt_array_filename} {gt_array_ds_name} {test_array_filename} {test_array_ds_name} {mask_array_filename} {mask_array_ds_name} {output_directory} {num_workers}"
#         os.system(
#             f"bsub -P cellmap -n {num_workers} -o {logfile_prefix}.o -e {logfile_prefix}.e python instance_segmentation_metrics.py {args}"
#         )
