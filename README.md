For preprocessing:
trained on the 8nm, evaluated on 4nm downsampled to 8nm
Run annotation-processing-utils cluster_submission (`/groups/cellmap/cellmap/ackermand/Programming/annotation-processing-utils/annotation_processing_utils/cli/cluster_submission.py`), pointing to the yamls here `/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/annotations/processing_yamls`


do the run inference, run mws, and run metrics. for run mws activate annotation_processing_utils_mws since that one requires numpy2 etc and didnt want to cause issues with annotation_processing_utils dask stuff etc.



