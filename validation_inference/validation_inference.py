from dacapo.store.create_store import (
    create_config_store,
    create_weights_store,
)
from dacapo.experiments import Run
from dacapo.predict import predict
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch
from pathlib import Path
import torch

from funlib.evaluate import rand_voi,detection_scores
import numpy as np
import sys
from funlib.segment.arrays import relabel
from funlib.geometry import Roi

from dacapo.experiments.datasplits.datasets.arrays import (ZarrArrayConfig, CropArrayConfig, IntensitiesArrayConfig)
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig

from dacapo.experiments.datasplits.datasets import RawGTDataset


def rvoi(X, Y):
    o = rand_voi(X, Y)
    return o["voi_split"] + o["voi_merge"]


def get_updated_validation_dataset(run):
    gt_config = ZarrArrayConfig(
        name="plasmodesmata",
        file_name=Path("/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5"),
        dataset="larger_validation_crop",
    )

    val_gt_config = CropArrayConfig(
        "val_gt", source_array_config=gt_config, roi=Roi((19952, 9736, 153344), (13464, 14064, 15104))
    )

    raw_config = ZarrArrayConfig(
        name="raw",
        file_name=Path("/nrs/stern/em_data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5"),
        dataset="em/fibsem-uint8/s0",
    )
    # We get an error without this, and will suggests having it as such https://cell-map.slack.com/archives/D02KBQ990ER/p1683762491204909
    raw_config = IntensitiesArrayConfig(
        name="raw", source_array_config=raw_config, min=0, max=255
    )
    validation_data_config = RawGTDatasetConfig("val", raw_config=raw_config, gt_config=val_gt_config)#, mask_config=mask_config
    return RawGTDataset(validation_data_config)


def validation_inference(run_name, iteration):
    #run_name = "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0"
    outpath = Path(f"/nrs/cellmap/ackermand/validation_inference_new_region/{run_name}")

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # create weights store and read weights
    weights_store = create_weights_store()
    post_processor = run.task.post_processor
    # gt = open_ds(
    #     "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0/validation.zarr",
    #     "inputs/val/gt",
    # ).data[:].astype(np.uint64)
    # relabel(gt, inplace = True)

    weights = weights_store.retrieve_weights(run, iteration)
    run.model.load_state_dict(weights.model)

    validation_dataset = get_updated_validation_dataset(run) #run.datasplit.validate[0]

    torch.backends.cudnn.benchmark = True
    run.model.eval()
    iteration_name = f"iteration_{iteration}"
    prediction_array_identifier = LocalArrayIdentifier(
        outpath / "predictions_larger_validation_crop.n5",
        iteration_name
    )

    predict(
        run.model,
        validation_dataset.raw,
        prediction_array_identifier,
        compute_context=LocalTorch(),
        output_roi=validation_dataset.gt.roi,
    )
    gt = validation_dataset.gt[validation_dataset.gt.roi].astype(np.uint64)
    relabel(gt, inplace = True)
    post_processor.set_prediction(prediction_array_identifier)
    for idx, parameters in enumerate(post_processor.enumerate_parameters()):
        output_array_identifier = LocalArrayIdentifier(outpath / "processed.n5", f"iteration_{iteration}/{parameters}")

        post_processed_array = post_processor.process(
            parameters, output_array_identifier
        )
        pred = post_processed_array.data[:]
        post_processed_array.attrs["rand_voi"] = rvoi(gt, pred)
        post_processed_array.attrs["rand_voi_include_background"] = rvoi(gt+1, pred+1)
        relabel(pred,inplace=True)
        post_processed_array.attrs["detection"] = detection_scores(gt, pred, matching_score="overlap", matching_threshold=1)
        post_processed_array.attrs["detection_iou"] = detection_scores(gt, pred, matching_score="iou", matching_threshold=0.1)



if __name__ == "__main__":
    validation_inference(sys.argv[1], sys.argv[2])