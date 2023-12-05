import os
from datetime import datetime
from pathlib import Path

for base_run_name in [
    "finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_more_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5",
    "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_more_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5",
    "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node_lr_5E-5",
    "finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_removed_dummy_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5",
]:
    for suffix in range(3):
        if not (
            "more_annotations" not in base_run_name  # then doesnt have three runs
            and suffix == 2
        ):
            run_name = f"{base_run_name}__{suffix}"
            dirname = datetime.now().strftime(f"%Y%m%d/{run_name}/%H%M%S")
            logdir = Path(f"logs/{dirname}")
            os.makedirs(logdir, exist_ok=True)

            for iteration in range(100000, 200001, 5000):  # 200000+1,5000):
                for crop in ["cyan", "purple"]:
                    for validation_or_test in ["validation", "test"]:
                        # if (
                        #     run_name
                        #     == "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node_lr_5E-5__1"
                        #     and iteration == 140000
                        #     and crop == "cyan"
                        #     and validation_or_test == "test"
                        # ):
                        os.system(
                            f"bsub -P cellmap -n 48 -o {logdir}/rusty_{iteration}_{crop}_{validation_or_test}.o -e {logdir}/rusty_{iteration}_{crop}_{validation_or_test}.e python validation_rusty_mws.py {run_name} {iteration} {crop} {validation_or_test}"
                        )
