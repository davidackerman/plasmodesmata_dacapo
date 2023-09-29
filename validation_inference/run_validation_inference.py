import os
from datetime import datetime
from pathlib import Path

for base_run_name in [
    "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_removed_dummy_annotations_unet_default_v2_no_dataset_predictor_node_lr_1E-4",
    #"finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_removed_dummy_annotations_unet_default_v2_no_dataset_predictor_node_lr_1E-4",
    #"finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_removed_dummy_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5",
    #"finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_removed_dummy_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5",
]:
    for suffix in range(2):
        run_name = f"{base_run_name}__{suffix}"
        dirname = datetime.now().strftime(f"%Y%m%d-%H%M%S-{run_name}")
        logdir = Path(f"logs/{dirname}")
        os.mkdir(logdir)
        for iteration in range(5000,5001):#200000+1,5000):
            os.system(f"bsub -P cellmap -q gpu_rtx -n 2 -gpu num=1 -o {logdir}/{iteration}.o -e {logdir}/{iteration}.e python validation_inference.py {run_name} {iteration}")
