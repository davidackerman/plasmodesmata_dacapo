# %%

# Write out annotations
from annotation_processing_utils.process.cylindrical_annotations import (
    CylindricalAnnotations,
)
import getpass

username = getpass.getuser()
organelle = "plasmodesmata"
dataset = "combined_healthy"


# save ca to pkl
# ca.save(f"./{dataset}_cylindrical_annotations.pkl")


# %%
# Save out dacapo runs
from dacapo.store.create_store import create_config_store
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig

combined_train_configs = []
combined_validate_configs = []
config_store = create_config_store()
annotation_name = "plasmodesmata"
for existing_dataset in ["jrc_22ak351-leaf-2l", "jrc_22ak351-leaf-3r"]:
    print(existing_dataset)
    # jsut retrieving the train/validate configs
    run_name = f"finetuned_3d_lsdaffs_weight_ratio_0.5_{existing_dataset}_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__0"
    run_config = config_store.retrieve_run_config(run_name)
    datasplit_config = run_config.datasplit_config

    combined_train_configs.extend(datasplit_config.train_configs)
    combined_validate_configs.extend(datasplit_config.validate_configs)

combined_datasplit_config = TrainValidateDataSplitConfig(
    name=f"{dataset}_{organelle}_all_training_points",
    train_configs=combined_train_configs,
    validate_configs=combined_validate_configs,
)
print("storing")
config_store.store_datasplit_config(combined_datasplit_config)
print("stored")
# %%
import time

print("creating run")
# lazy_results = []
for lsds_to_affs_weight_ratio in [0.5]:
    for batch_size in [2]:
        # print current time
        print(time.ctime())
        CylindricalAnnotations.create_combined_datasplit_dacapo_run(
            repetitions=2,
            lsds_to_affs_weight_ratio=lsds_to_affs_weight_ratio,
            batch_size=batch_size,
            validation_interval=1_000_000,
            datasplit_config=combined_datasplit_config,
        )
print("created")
# %%
# Visualize pipeline
# from dacapo.store.create_store import create_config_store
# from dacapo.experiments import Run

# config_store = create_config_store()
# run_config = config_store.retrieve_run_config(
#     f"finetuned_3d_lsdaffs_weight_ratio_0.5_{dataset}_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__0"
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

# # %%
# # import numpy as np
# # import yaml

# # # Example NumPy array
# # random_array = np.random.randint(0, 100, size=(100000, 3), dtype=np.int32)

# # # Convert to a list of tuples
# # tuple_list = [tuple(map(int, row)) for row in random_array]

# # # Convert to YAML format
# # yaml_output = yaml.dump(tuple_list, Dumper=yaml.Dumper)

# # print(yaml_output)

# # # %%
