import os
from datetime import datetime
from pathlib import Path
import yaml

yaml_path = "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/validation_and_test/yamls/combined_yamls/jrc_22ak351-leaf-2l_2023-12-15.yml"
with open(yaml_path, "r") as stream:
    yml = yaml.safe_load(stream)

iterations_start, iterations_end, iterations_step = yml["iterations"]
for run in yml["runs"]:
    dirname = datetime.now().strftime(f"%Y%m%d/{run}/%H%M%S")
    logdir = Path(
        f"/nrs/cellmap/ackermand/logs/plasmodesmata_dacapo/validation_and_test/{dirname}"
    )
    os.makedirs(logdir, exist_ok=True)
    for roi_info in yml["rois"]:
        roi_name = roi_info["name"]
        for iteration in range(iterations_start, iterations_end + 1, iterations_step):
            os.system(
                f"bsub -P cellmap -q gpu_rtx -n 1 -gpu num=1 -o {logdir}/inference_{iteration}_{roi_name}.o -e {logdir}/inference_{iteration}_{roi_name}.e python validation_inference.py {yaml_path} {run} {iteration} {roi_name}"
            )
