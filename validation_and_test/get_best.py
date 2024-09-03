import json
from dacapo.store.create_store import (
    create_config_store,
)
from dacapo.experiments import Run
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    columns=[
        "run",
        "iteration",
        "parameter",
        "full_path",
        "rand_voi",
        "rand_voi_bkgd",
        "detection_f1",
    ]
)


runs = [
    "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0",
    "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__1",
    "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0",
    "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1",
]
for run_name in runs:
    for iteration in range(5000, 200000 + 1, 5000):
        for idx, bias in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
            parameter = f"WatershedPostProcessorParameters(id={idx}, bias={bias})"
            dir_name = f"/nrs/cellmap/ackermand/validation_inference/{run_name}/processed.n5/iteration_{iteration}/{parameter}"
            file_name = f"{dir_name}/attributes.json"
            with open(file_name) as f:
                data = json.load(f)
            detection = data["detection"]
            f1 = (
                2
                * detection["tp"]
                / (2 * detection["tp"] + detection["fp"] + detection["fn"])
            )
            row = [
                run_name,
                iteration,
                parameter,
                dir_name,
                data["rand_voi"],
                data["rand_voi_include_background"],
                f1,
            ]
            df.loc[len(df.index)] = row

for run in df["run"].unique():
    for metric in ["rand_voi", "rand_voi_bkgd", "detection_f1"]:
        df_run = df[df["run"] == run]
        df_run.reset_index(inplace=True)
        if "voi" in metric:
            best_idx = df_run[metric].idxmin()
        else:
            best_idx = df_run[metric].idxmax()
        row = df_run.iloc[[best_idx]]
        print(
            f'{row["run"].values[0]} best {metric}: {row[metric].values[0]} at location {row["full_path"].values[0]}'
        )