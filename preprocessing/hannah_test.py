#%%
#  Write out annotations
from annotation_processing_utils.process.cylindrical_annotations import (
    CylindricalAnnotations,
)
import getpass

username = getpass.getuser()
organelle = "nuclear_pore"
dataset = "jrc_sum159-1"
radius = 7
ca = CylindricalAnnotations(
    organelle=organelle,
    training_validation_test_roi_info_yaml=f"./hannah_test.yaml",
    output_mask_zarr=f"/nrs/cellmap/{username}/test_cellmap_experiments/hannah_test/cellmap/{organelle}/annotation_intersection_masks.zarr",
    output_gt_zarr=f"/nrs/cellmap/{username}/test_cellmap_experiments/hannah_test/cellmap/{organelle}/annotations_as_cylinders.zarr",
    output_training_points_zarr=f"/nrs/cellmap/{username}/test_cellmap_experiments/hannah_test/{organelle}/training_points.zarr",
    output_annotations_directory=f"/nrs/cellmap/{username}/test_cellmap_experiments/hannah_test/{organelle}/neuroglancer_annotations",
    dataset=dataset,
    radius=radius,
)
ca.standard_processing()

# %%
