from dacapo.store.create_store import (
    create_config_store,
    create_weights_store,
)
from dacapo.experiments import Run
from predict_with_write_size import predict_with_write_size
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch
from pathlib import Path
import torch

import numpy as np
import sys
from funlib.geometry import Roi
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    CropArrayConfig,
    IntensitiesArrayConfig,
)
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits.datasets import RawGTDataset
from model import Model
import pandas as pd
import logging
import shutil
import yaml

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.NOTSET,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_validation_and_test_rois(
    annotation_path, box_start, box_end, split_dimension, voxel_size=8
):
    box_start = np.ceil(np.array(box_start) / 8).astype(int)
    box_end = np.floor(np.array(box_end) / 8).astype(int)
    if split_dimension is None:
        # then we decided it is too small to split and will just use for validation
        return {
            "validation": Roi(
                box_start[::-1] * voxel_size, (box_end - box_start)[::-1] * voxel_size
            )
        }

    best_box_first_half = box_end.copy()
    best_box_second_half = box_start.copy()
    best_score = np.inf

    df = pd.read_csv(annotation_path)
    pd_starts = np.array([df["start x (nm)"], df["start y (nm)"], df["start z (nm)"]]).T
    pd_ends = np.array([df["end x (nm)"], df["end y (nm)"], df["end z (nm)"]]).T
    pd_centers = np.round(((pd_starts + pd_ends) / 2)).astype(int)

    # check pd centers are within region
    valid_pds = (
        (pd_centers[:, 0] / voxel_size >= box_start[0])
        & (pd_centers[:, 0] / voxel_size <= box_end[0])
        & (pd_centers[:, 1] / voxel_size >= box_start[1])
        & (pd_centers[:, 1] / voxel_size <= box_end[1])
        & (pd_centers[:, 2] / voxel_size >= box_start[2])
        & (pd_centers[:, 2] / voxel_size <= box_end[2])
    )
    pd_centers = pd_centers[valid_pds, :]

    for box_split in range(box_start[split_dimension], box_end[split_dimension]):
        first_half = np.sum(pd_centers[:, split_dimension] < box_split * voxel_size)
        second_half = np.sum(
            pd_centers[:, split_dimension] >= (box_split + 145) * voxel_size
        )
        if second_half > 0:
            ratio = first_half / second_half

            if np.abs(1 - ratio) < best_score:
                best_score = np.abs(1 - ratio)
                best_box_first_half[split_dimension] = box_split
                best_box_second_half[split_dimension] = box_split + 145
    # swap axes to get in z,y,x
    box_start = box_start[::-1]
    box_end = box_end[::-1]
    best_box_first_half = best_box_first_half[::-1]
    best_box_second_half = best_box_second_half[::-1]
    validation_and_test_roi_dict = {
        "validation": Roi(
            box_start * voxel_size, (best_box_first_half - box_start) * voxel_size
        ),
        "test": Roi(
            best_box_second_half * voxel_size,
            (box_end - best_box_second_half) * voxel_size,
        ),
    }
    return validation_and_test_roi_dict


def create_model(architecture):
    head = torch.nn.Conv3d(72, 19, kernel_size=1)
    return Model(architecture, head, eval_activation=torch.nn.Sigmoid())


def get_updated_validation_dataset(roi, dataset):
    gt_config = ZarrArrayConfig(
        name="plasmodesmata",
        file_name=Path(f"/nrs/cellmap/ackermand/cellmap/leaf-gall/{dataset}.n5"),
        dataset="plasmodesmata_as_cylinders",
    )

    val_gt_config = CropArrayConfig("val_gt", source_array_config=gt_config, roi=roi)

    raw_config = ZarrArrayConfig(
        name="raw",
        file_name=Path(f"/nrs/stern/em_data/{dataset}/{dataset}.n5"),
        dataset="em/fibsem-uint8/s0",
    )
    # We get an error without this, and will suggests having it as such https://cell-map.slack.com/archives/D02KBQ990ER/p1683762491204909
    raw_config = IntensitiesArrayConfig(
        name="raw", source_array_config=raw_config, min=0, max=255
    )
    validation_data_config = RawGTDatasetConfig(
        "val", raw_config=raw_config, gt_config=val_gt_config
    )  # , mask_config=mask_config
    return RawGTDataset(validation_data_config)


def validation_inference(path_to_yml, run_name, iteration, roi_name):
    with open(path_to_yml, "r") as stream:
        yml = yaml.safe_load(stream)
    dataset = yml["dataset"]

    outpath = Path(
        f"/nrs/cellmap/ackermand/predictions/{dataset}/{dataset}.n5/predictions/"
    )

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)
    run.model = create_model(run.architecture)
    # create weights store and read weights
    weights_store = create_weights_store()
    weights = weights_store.retrieve_weights(run, iteration)
    run.model.load_state_dict(weights.model)

    torch.backends.cudnn.benchmark = True
    run.model.eval()
    iteration_name = f"iteration_{iteration}"
    for whole_roi_info in yml["rois"]:
        if str(whole_roi_info["name"]) == str(roi_name):
            split_roi_info = get_validation_and_test_rois(
                whole_roi_info["path"],
                whole_roi_info["start"],
                whole_roi_info["end"],
                split_dimension=whole_roi_info["split_dimension"],
            )

            for roi_type, roi in split_roi_info.items():
                shutil.rmtree(
                    outpath
                    / roi_type
                    / run_name
                    / f'{whole_roi_info["name"]}.n5/{iteration_name}',
                    ignore_errors=True,
                )
                validation_dataset = get_updated_validation_dataset(roi, dataset)
                prediction_array_identifier = LocalArrayIdentifier(
                    outpath / roi_type / run_name / f'{whole_roi_info["name"]}.n5',
                    iteration_name,
                )
                predict_with_write_size(
                    run.model,
                    validation_dataset.raw,
                    prediction_array_identifier,
                    compute_context=LocalTorch(),
                    output_roi=validation_dataset.gt.roi,
                    write_size=[108 * 8, 108 * 8, 108 * 8, 9],
                )


if __name__ == "__main__":
    validation_inference(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
