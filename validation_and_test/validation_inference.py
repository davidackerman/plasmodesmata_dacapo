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

from funlib.evaluate import rand_voi, detection_scores
import numpy as np
import sys
from funlib.segment.arrays import relabel
from funlib.geometry import Roi, Coordinate

from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    CropArrayConfig,
    IntensitiesArrayConfig,
)
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig

from dacapo.experiments.datasplits.datasets import RawGTDataset
from funlib.persistence import open_ds, Array

from model import Model
import warnings
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.NOTSET,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def rvoi(X, Y):
    o = rand_voi(X, Y)
    return o["voi_split"] + o["voi_merge"]


def create_model(architecture):
    head = torch.nn.Conv3d(72, 19, kernel_size=1)
    return Model(architecture, head, eval_activation=torch.nn.Sigmoid())


def get_updated_validation_dataset(run, roi):
    gt_config = ZarrArrayConfig(
        name="plasmodesmata",
        file_name=Path(
            "/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5"
        ),
        dataset="plasmodesmata_as_cylinders",
    )

    val_gt_config = CropArrayConfig("val_gt", source_array_config=gt_config, roi=roi)

    raw_config = ZarrArrayConfig(
        name="raw",
        file_name=Path("/nrs/stern/em_data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5"),
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


def validation_inference(run_name, iteration):
    # run_name = "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0"
    outpath = Path(
        f"/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5/predictions/"
    )

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)
    run.model = create_model(run.architecture)
    # create weights store and read weights
    weights_store = create_weights_store()
    # post_processor = run.task.post_processor
    # gt = open_ds(
    #     "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0/validation.zarr",
    #     "inputs/val/gt",
    # ).data[:].astype(np.uint64)
    # relabel(gt, inplace = True)

    weights = weights_store.retrieve_weights(run, iteration)
    run.model.load_state_dict(weights.model)

    torch.backends.cudnn.benchmark = True
    run.model.eval()
    iteration_name = f"iteration_{iteration}"

    # roi_info = [
    #     ("cyan", "validation", Roi([82296, 22800, 24704][::-1],[ 3856, 15696,  9072][::-1])),
    #     ("cyan", "test", Roi([82296, 22800, 34936][::-1],[ 3856, 15696,  7344][::-1])),
    #     ("purple", "validation", Roi([209176,  25016,  76184][::-1],[ 8416, 12104, 10784][::-1])),
    #     ("purple", "test", Roi([218752,  25016,  76184][::-1],[ 8600, 12104, 10784][::-1])),
    # ]

    roi_info = [
        (
            "cyan",
            "validation",
            Roi([82304, 22808, 24712][::-1], [3832, 15672, 9064][::-1]),
        ),
        ("cyan", "test", Roi([82304, 22808, 34936][::-1], [3832, 15672, 7328][::-1])),
        (
            "purple",
            "validation",
            Roi([209184, 25024, 76192][::-1], [8408, 12080, 10760][::-1]),
        ),
        (
            "purple",
            "test",
            Roi([218752, 25024, 76192][::-1], [8584, 12080, 10760][::-1]),
        ),
    ]
    for roi_name, roi_type, roi in roi_info:
        validation_dataset = get_updated_validation_dataset(run, roi)
        prediction_array_identifier = LocalArrayIdentifier(
            outpath / roi_type / run_name / f"{roi_name}.n5", iteration_name
        )

        predict_with_write_size(
            run.model,
            validation_dataset.raw,
            prediction_array_identifier,
            compute_context=LocalTorch(),
            output_roi=validation_dataset.gt.roi,
            write_size=[108 * 8, 108 * 8, 108 * 8, 9],
        )
        # break
        # warnings.warn("getting gt")
        # gt = validation_dataset.gt[validation_dataset.gt.roi].astype(np.uint64)
        # #gt = validation_dataset.gt.to_ndarray(roi).astype(np.uint64)
        # warnings.warn(f"got gt {gt.size}")
        # warnings.warn("relabeling")
        # relabel(gt, inplace=True)
        # warnings.warn("relabeled")
        # warnings.warn("setting prediction")
        # post_processor.set_prediction(prediction_array_identifier)
        # for idx, parameters in enumerate(post_processor.enumerate_parameters()):
        #     o = f"{idx},{parameters}"
        #     warnings.warn(o)
        #     if idx==2:
        #         output_array_identifier = LocalArrayIdentifier(
        #             outpath / "processed.n5", f"iteration_{iteration}/{parameters}"
        #         )
        #         warnings.warn("processing")
        #         post_processed_array = post_processor.process(
        #             parameters, output_array_identifier
        #         )
        #         warnings.warn("processed")
        #         pred = post_processed_array.data[:]
        #         # post_processed_array.attrs["rand_voi"] = rvoi(gt, pred)
        #         # post_processed_array.attrs["rand_voi_include_background"] = rvoi(
        #         #     gt + 1, pred + 1
        #         # )
        #         relabel(pred, inplace=True)
        #         warnings.warn("detecting")
        #         post_processed_array.attrs["detection"] = detection_scores(
        #             gt, pred, matching_score="overlap", matching_threshold=1
        #         )
        #         post_processed_array.attrs["detection_iou"] = detection_scores(
        #             gt, pred, matching_score="iou", matching_threshold=0.1
        #         )


if __name__ == "__main__":
    validation_inference(sys.argv[1], sys.argv[2])
