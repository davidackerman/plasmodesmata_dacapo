import os
from datetime import datetime
from pathlib import Path
import yaml

with open(
    "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/validation_and_test/metrics_yamls/jrc_22ak351-leaf-3m_2023-12-04.yml",
    "r",
) as stream:
    yml = yaml.safe_load(stream)

dataset = yml["dataset"]
gt_array_filename = yml["gt_array"]["filename"]
gt_array_ds_name = yml["gt_array"]["ds_name"]
mask_array_filename = yml["mask_array"]["filename"]
mask_array_ds_name = yml["mask_array"]["ds_name"]

iterations_start, iterations_end, iterations_step = yml["iterations"]
date_and_time = datetime.now().strftime(f"%Y%m%d_%H%M%S")
date_and_time = "NO_MASK"
# date_and_time = "20231204_232852"
num_workers = 10
for run in yml["runs"]:
    for iteration in range(iterations_start, iterations_end + 1, iterations_step):
        for roi_name in yml["roi_names"]:
            for postprocessing_suffix in yml["postprocessing_suffixes"]:
                for validation_or_test in ["validation", "test"]:
                    test_array = yml["test_array"]
                    test_array_filename = f'{test_array["base_filename"]}/processed/{validation_or_test}/{run}/{roi_name}.n5'
                    test_array_ds_name = f"iteration_{iteration}{postprocessing_suffix}"
                    output_directory = f"/nrs/cellmap/ackermand/validation_and_testing_scores/{date_and_time}/{validation_or_test}/{run}/{roi_name}/iteration_{iteration}{postprocessing_suffix}"
                    if (
                        run
                        == "finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_removed_dummy_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5__0"
                        and iteration == 105000
                        and roi_name == "cyan"
                        and validation_or_test == "validation"
                    ):
                        log_dir = datetime.now().strftime(f"logs/%Y%m%d/{run}/%H%M%S")
                        os.makedirs(log_dir, exist_ok=True)
                        logfile_prefix = f"{log_dir}/metrics_{iteration}_{roi_name}_{validation_or_test}_{postprocessing_suffix}"
                        args = f"{dataset} {gt_array_filename} {gt_array_ds_name} {test_array_filename} {test_array_ds_name} {mask_array_filename} {mask_array_ds_name} {output_directory} {num_workers}"
                        os.system(
                            f"bsub -P cellmap -n {num_workers} -o {logfile_prefix}.o -e {logfile_prefix}.e python instance_segmentation_metrics.py {args}"
                        )
