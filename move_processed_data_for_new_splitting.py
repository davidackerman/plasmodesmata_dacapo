# %%
from annotation_processing_utils.process.training_validation_test_roi_calculator import (
    TrainingValidationTestRoiCalculator,
)
import yaml
import os
import shutil

dataset_name = "jrc_22ak351-leaf-2lb"
yaml_path = f"/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/processing_yamls/{dataset_name}.yaml"
with open(yaml_path, "r") as f:
    yaml_data = yaml.safe_load(f)

roi_splitter = TrainingValidationTestRoiCalculator(
    yaml_data["analysis_info"][0]["training_validation_test_roi_yaml"]
)
roi_splitter.get_training_validation_test_rois()
rois_dict = roi_splitter.rois_dict
print(rois_dict)

for run in yaml_data["runs"]:
    for validation_or_test in rois_dict.keys():
        for crop in rois_dict[validation_or_test].keys():
            if validation_or_test in ["test"]:
                validation_inference_path = f"/nrs/cellmap/ackermand/predictions/leaf-gall/{dataset_name}/{dataset_name}.zarr/predictions/{dataset_name}/validation/{run}/{crop}"
                validation_processed_path = f"/nrs/cellmap/ackermand/predictions/leaf-gall/{dataset_name}/{dataset_name}.zarr/processed/{dataset_name}/validation/{run}/{crop}"
                validation_scores_path = f"/nrs/cellmap/ackermand/validation_and_testing_scores/leaf-gall/{dataset_name}/{dataset_name}/validation/{run}/{crop}"

                test_inference_path = f"/nrs/cellmap/ackermand/predictions/leaf-gall/{dataset_name}/{dataset_name}.zarr/predictions/{dataset_name}/test/{run}/"
                test_processed_path = f"/nrs/cellmap/ackermand/predictions/leaf-gall/{dataset_name}/{dataset_name}.zarr/processed/{dataset_name}/test/{run}/"
                test_scores_path = f"/nrs/cellmap/ackermand/validation_and_testing_scores/leaf-gall/{dataset_name}/{dataset_name}/test/{run}/"
                # os.makedirs(test_inference_path, exist_ok=True)
                # os.makedirs(test_processed_path, exist_ok=True)
                # os.makedirs(test_scores_path, exist_ok=True)
                # shutil.move(validation_inference_path, test_inference_path)
                # shutil.move(validation_processed_path, test_processed_path)
                # shutil.move(validation_scores_path, test_scores_path)


# %%
