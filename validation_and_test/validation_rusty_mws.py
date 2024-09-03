import rusty_mws
import sys
import logging

logger: logging.Logger = logging.getLogger(name=__name__)
if __name__ == "__main__":
    run_name = sys.argv[1]
    iteration = sys.argv[2]
    crop = sys.argv[3]
    validation_or_test = sys.argv[4]
    dataset = sys.argv[5]
    base_path = f"/nrs/cellmap/ackermand/predictions/{dataset}/{dataset}.n5"
    for adj_bias, lr_bias in [(0.5, -1.2)]:  # ,(0.1,-1.2)]:
        iteration_name = f"iteration_{iteration}"
        log_dir = f"/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/validation_and_test/logs/daisy_logs/{dataset}/rusty_mws_logs/{validation_or_test}/{run_name}/{crop}/{iteration}/"
        affs_file = f"{base_path}/predictions/{validation_or_test}/{run_name}/{crop}.n5"
        outfile = f"{base_path}/processed/{validation_or_test}/{run_name}/{crop}.n5"
        filter_val = 0.5
        lr_bias_ratio = -0.08
        pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
            # sample_name="test",
            affs_file=affs_file,
            affs_dataset=iteration_name,
            fragments_file=outfile,
            fragments_dataset=f"{iteration_name}_filter_val_{filter_val}_lrb_ratio_{lr_bias_ratio}_frags",
            seg_file=outfile,
            seg_dataset=f"{iteration_name}_filter_val_{filter_val}_lrb_ratio_{lr_bias_ratio}_adj_{adj_bias}_lr_{lr_bias}_segs",
            db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017",
            db_name="rusty_mws_ackermand",
            lr_bias_ratio=lr_bias_ratio,
            adj_bias=adj_bias,
            lr_bias=lr_bias,
            nworkers_frags=62,
            nworkers_lut=62,
            nworkers_supervox=62,
            filter_val=filter_val,
            neighborhood=[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [3, 0, 0],
                [0, 3, 0],
                [0, 0, 3],
                [9, 0, 0],
                [0, 9, 0],
                [0, 0, 9],
            ],
            log_dir=log_dir,
        )
        success = pp.run_pred_segmentation_pipeline()
        if success:
            print(
                f"completed ({crop}, {validation_or_test} {filter_val}, {adj_bias}, {lr_bias}) all tasks successfully!"
            )
        else:
            print(
                f"failed ({crop}, {validation_or_test} {filter_val}, {adj_bias}, {lr_bias}) with status success Some task failed."
            )
