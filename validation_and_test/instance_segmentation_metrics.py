import sys
import logging
from funlib.persistence import open_ds
from utils import *
from datetime import datetime
from pathlib import Path

logger: logging.Logger = logging.getLogger(name=__name__)
if __name__ == "__main__":
    dataset = sys.argv[1]
    gt_array_filename = sys.argv[2]
    gt_array_ds_name = sys.argv[3]
    test_array_filename = sys.argv[4]
    test_array_ds_name = sys.argv[5]
    mask_array_filename = sys.argv[6]
    mask_array_ds_name = sys.argv[7]
    output_directory = sys.argv[8]
    num_workers = sys.argv[9]

    gt_array = open_ds(gt_array_filename, gt_array_ds_name)
    test_array = open_ds(test_array_filename, test_array_ds_name)
    mask_array = open_ds(mask_array_filename, mask_array_ds_name)
    dirname = datetime.now().strftime(
        f"metrics/%Y%m%d/{test_array_filename}/{test_array_ds_name}/%H%M%S"
    )
    log_dir = Path(f"logs/{dirname}")

    os.makedirs(log_dir, exist_ok=True)
    isos = InstanceSegmentationOverlapAndScorer(
        gt_array,
        test_array,
        mask_array,
        output_directory,
        log_dir,
        int(num_workers),
    )
    isos.process()
