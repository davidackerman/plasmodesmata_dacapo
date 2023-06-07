from funlib.geometry import Roi

from dacapo.store.create_store import create_config_store,create_config_store,create_weights_store
from dacapo.experiments import Run
from dacapo.predict import predict
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch
from pathlib import Path
import torch


if __name__ == "__main__":

    run_name = "finetuned_3d_lsdaffs_plasmodesmata_upsample-unet_default_v2__0"
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # create weights store and read weights
    weights_store = create_weights_store()
    weights = weights_store.retrieve_weights(run, 165000)
    weights_store._load_best(run, "val/voi")

    validation_dataset = run.datasplit.validate[0]
    output_rois = [Roi(validation_dataset.gt.roi.begin, 3*[i]) for i in [54*8, 55*8, 108*8, 216*8, 217*8, 324*8]]
    output_rois.append(validation_dataset.gt.roi)  # this is not a cube, it is 200x200x300 voxels
    output_rois.append(Roi(validation_dataset.gt.roi.begin, [54*8, 54*8, 60*8]))  # make another non-cube

    torch.backends.cudnn.benchmark = True
    run.model.eval()
    output_path = Path("./temp.n5")
    for output_roi in output_rois:
        output_dataset = f"{output_roi.shape[0]}x{output_roi.shape[1]}x{output_roi.shape[2]}"
        prediction_array_identifier = LocalArrayIdentifier(output_path, output_dataset)
        predict(
                run.model,
                validation_dataset.raw,
                prediction_array_identifier,
                compute_context=LocalTorch(),
                output_roi=output_roi,
            )
