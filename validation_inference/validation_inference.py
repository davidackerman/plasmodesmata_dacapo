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
from funlib.persistence import open_ds
import sys
from funlib.segment.arrays import relabel


def rvoi(X, Y):
    o = rand_voi(X, Y)
    return o["voi_split"] + o["voi_merge"]


def validation_inference(run_name, iteration):
    #run_name = "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0"
    outpath = Path(f"/nrs/cellmap/ackermand/validation_inference/{run_name}")

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

    validation_dataset = run.datasplit.validate[0]

    torch.backends.cudnn.benchmark = True
    run.model.eval()
    iteration_name = f"iteration_{iteration}"
    prediction_array_identifier = LocalArrayIdentifier(
        outpath / "predictions.n5",
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