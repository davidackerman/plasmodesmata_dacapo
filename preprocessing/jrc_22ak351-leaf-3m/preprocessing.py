#!/usr/bin/env python
# coding: utf-8

# ## create rasterized datasets

# # NOTE: a lot of new regions were added as validation, so were not included initially. This is why we had 4427 annotations, but with validation we have 6059: we had to write out the annotations again for validation purposes, even though the new updates werent included for training!
# 

# In[6]:


# everything created after september wasnt included in training and were solely included for validation
import pandas as pd

annotation_csvs = [
    "../annotations/jrc_22ak351-leaf-3m/annotations_20230510_114340_dummy_removed.csv",  # original annotations (yellow), 0
    "../annotations/jrc_22ak351-leaf-3m//annotations_20230829_173628.csv",  # new region 1 (big yellow box, white annotations)
    "../annotations/jrc_22ak351-leaf-3m//annotations_20230929_115330.csv",  # new region 2 (cyan)
    "../annotations/jrc_22ak351-leaf-3m//annotations_20230929_115745.csv",  # new region 3 (purple)
    "../annotations/jrc_22ak351-leaf-3m//annotations_20230929_115914.csv",  # new regions 4 (pink) and 5 (green)
    # with the following you get to 6059:
    # "../annotations/jrc_22ak351-leaf-3m//annotations_20231206_161924.csv",  # new region 6
    # "../annotations/jrc_22ak351-leaf-3m//annotations_20231206_162425.csv",  # new region 7
    # "../annotations/jrc_22ak351-leaf-3m//annotations_20231206_162508.csv",  # new region 8
    # "../annotations/jrc_22ak351-leaf-3m//annotations_20231206_162618.csv",  # new region 9
    # "../annotations/jrc_22ak351-leaf-3m//annotations_20231206_162718.csv",  # new region 10
    # "../annotations/jrc_22ak351-leaf-3m//annotations_20231206_162754.csv",  # new region 11
]
dfs = []
for annotation in annotation_csvs:
    dfs.append(pd.read_csv(annotation))
df = pd.concat(dfs)
df


# In[23]:


import pandas as pd
import numpy as np
from funlib.geometry import Roi
from tqdm import tqdm
from annotation_processing_utils.process.cylindrical_annotations import (
    CylindricalAnnotations,
)

# all annotations: https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/2023-09-29.140842.json
annotation_csvs = [
    "annotations/annotations_20230510_114340_dummy_removed.csv",  # original annotations (yellow), 0
    "annotations/annotations_20230829_173628.csv",  # new region 1 (big yellow box, white annotations)
    "annotations/annotations_20230929_115330.csv",  # new region 2 (cyan)
    "annotations/annotations_20230929_115745.csv",  # new region 3 (purple)
    "annotations/annotations_20230929_115914.csv",  # new regions 4 (pink) and 5 (green)
    "annotations/annotations_20231206_161924.csv",  # new region 6
    "annotations/annotations_20231206_162425.csv",  # new region 7
    "annotations/annotations_20231206_162508.csv",  # new region 8
    "annotations/annotations_20231206_162618.csv",  # new region 9
    "annotations/annotations_20231206_162718.csv",  # new region 10
    "annotations/annotations_20231206_162754.csv",  # new region 11
]

# 2 (cyan) bounding box:
box_start = np.array([10287, 2850, 3088])
box_end = np.array([10769, 4812, 5285])
cyan_bounding_box = Roi(box_start, box_end - box_start)

# 3 (purple) bounding box:
box_start = np.array([26147, 3127, 9523])
box_end = np.array([28419, 4640, 10871])
purple_bounding_box = Roi(box_start, box_end - box_start)

# densely annotated validation region:
original_validation_bounding_box = Roi(
    [27400, 2000, 5300], [300, 200, 200]
)  # even though attributes says 5100

preprocess = CylindricalAnnotations(
    annotation_csvs=annotation_csvs,
    validation_or_test_rois=[
        cyan_bounding_box,
        purple_bounding_box,
        original_validation_bounding_box,
    ],
)
preprocess.extract_annotation_information()
preprocess.write_annotations_as_cylinders_and_get_intersections()
preprocess.write_intersection_mask()
# seems to work https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/2023-09-30.025524.json


# In[5]:


import matplotlib.pyplot as plt

plt.hist(np.linalg.norm(preprocess.pd_starts - preprocess.pd_ends, axis=1) * 8)


# In[2]:


preprocess.remove_validation_or_test_annotations_from_training()


# In[9]:


preprocess.write_out_removed_annotations()
# preprocess.visualize_removed_annotations(original_validation_bounding_box)


# In[ ]:


preprocess.removed_ids


# # TODO: figure out proper splitting of crops for validation and test. will be the cyan and purple ones

# In[3]:


import numpy as np
import pandas as pd
from funlib.geometry import Roi

# https://cell-map.slack.com/archives/C04RT6EN59A/p1686696980205189
# you add 72 on the ends for context, so we want to make sure that no context is shared so there should be a gap >72*2=144
# cyan bounding box:
box_start = np.array(
    [10287 + 1, 2850 + 1, 3088 + 1]
)  # add one and subtract 2 on next line because the rois aren't necessarily voxel aligned so need to be conservative
box_end = np.array([10769 - 2, 4812 - 2, 5285 - 2])
box_center = (box_start + box_end) / 2

df = pd.read_csv("annotations_20230929_115330.csv")  # cyan
pd_starts = np.array([df["start x (nm)"], df["start y (nm)"], df["start z (nm)"]]).T / 8
pd_ends = np.array([df["end x (nm)"], df["end y (nm)"], df["end z (nm)"]]).T / 8
pd_centers = np.round(((pd_starts + pd_ends) * 8 / 2)).astype(int)

best_score = np.inf
best_box_split = -1
best_box_first_half = box_end.copy()
best_box_second_half = box_start.copy()
dim = 2
for box_split in range(box_start[dim], box_end[dim]):
    first_half = np.sum(pd_centers[:, dim] < box_split * 8)
    second_half = np.sum(pd_centers[:, dim] >= (box_split + 145) * 8)
    ratio = first_half / second_half

    if np.abs(1 - ratio) < best_score:
        best_score = np.abs(1 - ratio)
        best_ratio = ratio
        best_box_split = box_split
        best_first_half = first_half
        best_second_half = second_half
        best_box_first_half[dim] = box_split
        best_box_second_half[dim] = box_split + 145
print(f'"cyan", "validation", {box_start*8},{best_box_first_half*8-box_start*8}')
print(f'"cyan", "test", {best_box_second_half*8},{box_end*8-best_box_second_half*8}')


# purple bounding box:
box_start = np.array(
    [26147 + 1, 3127 + 1, 9523 + 1]
)  # add one and subtract 2 on next line because the rois aren't necessarily voxel aligned so need to be conservative
box_end = np.array([28419 - 2, 4640 - 2, 10871 - 2])
box_center = (box_start + box_end) / 2

df = pd.read_csv("annotations_20230929_115745.csv")  # new region 3 (purple)
pd_starts = np.array([df["start x (nm)"], df["start y (nm)"], df["start z (nm)"]]).T / 8
pd_ends = np.array([df["end x (nm)"], df["end y (nm)"], df["end z (nm)"]]).T / 8
pd_centers = np.round(((pd_starts + pd_ends) * 8 / 2)).astype(int)

best_score = np.inf
best_box_split = -1
dim = 0
best_box_first_half = box_end.copy()
best_box_second_half = box_start.copy()
for box_split in range(box_start[dim], box_end[dim]):
    first_half = np.sum(pd_centers[:, dim] < box_split * 8)
    second_half = np.sum(pd_centers[:, dim] >= (box_split + 145) * 8)
    ratio = first_half / second_half

    if np.abs(1 - ratio) < best_score:
        best_score = np.abs(1 - ratio)
        best_ratio = ratio
        best_box_split = box_split
        best_first_half = first_half
        best_second_half = second_half
        box_final = box_start
        best_box_first_half[dim] = box_split
        best_box_second_half[dim] = box_split + 145

print(f'"purple", "validation", {box_start*8},{best_box_first_half*8-box_start*8}')
print(f'"purple", "test", {best_box_second_half*8},{box_end*8-best_box_second_half*8}')


# In[107]:


from funlib.geometry import Roi


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


import yaml

with open(
    f"/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/validation_and_test/yamls/validation_inference/jrc_22ak351-leaf-3m_2023-12-06.yml",
    "r",
) as stream:
    yml = yaml.safe_load(stream)


for whole_roi_info in yml["rois"]:
    split_roi_info = get_validation_and_test_rois(
        whole_roi_info["path"],
        whole_roi_info["start"],
        whole_roi_info["end"],
        split_dimension=whole_roi_info["split_dimension"],
    )

    for roi_type, roi in split_roi_info.items():
        print(roi_type, roi, whole_roi_info["name"])

# "purple", "validation", [209184  25024  76192],[ 8408 12080 10760]
# "purple", "test", [218752  25024  76192],[ 8584 12080 10760]


# In[60]:


yml


# In[31]:


np.floor(box_end / 8, dtype=int)


# ## new training region boxes

# 

# In[1]:


import pandas as pd
import numpy as np
from funlib.geometry import Roi, Coordinate
from funlib.persistence import prepare_ds

# # https://cell-map.slack.com/archives/D05KB53AN82/p1693413725612029
# start_2l = np.array((13084, 1018, 4965))[::-1] * 8
# end_2l = np.array((14814, 2018, 6136))[::-1] * 8
# roi_2l = Roi(start_2l, end_2l - start_2l)

# output_ds = prepare_ds(
#     "/nrs/cellmap/ackermand/forGrace/jrc_22ak351-leaf-2l.n5",
#     "annotation_box_1",
#     total_roi=roi_2l,
#     voxel_size=np.array([8, 8, 8]),
#     dtype=np.uint8,
#     write_size=Coordinate(np.array([64, 64, 64]) * 128),
# )

# old
# # https://cell-map.slack.com/archives/C04N9JUFQK1/p1692812572002639
# start_2l = np.array((5700, 3750, 2282))[::-1] * 8
# end_2l = np.array((8200, 5250, 3282))[::-1] * 8
# roi_2l = Roi(start_2l, end_2l - start_2l)
# output_ds = prepare_ds(
#     "/nrs/cellmap/ackermand/forXin/jrc_22ak351-leaf-2l.n5",
#     "annotation_box_1",
#     total_roi=roi_2l,
#     voxel_size=np.array([8, 8, 8]),
#     dtype=np.uint8,
#     write_size=Coordinate(np.array([64, 64, 64]) * 128),
# )

# https://cell-map.slack.com/archives/D05M1PHTV32/p1694097621746439
start_2l = np.array((5998, 3185, 1935))[::-1] * 8
end_2l = np.array((8098, 5405, 3564))[::-1] * 8
roi_2l = Roi(start_2l, end_2l - start_2l)
output_ds = prepare_ds(
    "/nrs/cellmap/ackermand/forXin/jrc_22ak351-leaf-2l.n5",
    "annotation_box_1",
    total_roi=roi_2l,
    voxel_size=np.array([8, 8, 8]),
    dtype=np.uint8,
    write_size=Coordinate(np.array([64, 64, 64]) * 128),
)

# # https://cell-map.slack.com/archives/D05KB53AN82/p1693414480454379
# start_3r = np.array((9000, 2123, 3329))[::-1] * 8
# end_3r = np.array((10000, 3000, 5329))[::-1] * 8
# roi_3r = Roi(start_3r, end_3r - start_3r)

# output_ds = prepare_ds(
#     "/nrs/cellmap/ackermand/forGrace/jrc_22ak351-leaf-3r.n5",
#     "annotation_box_1",
#     total_roi=roi_3r,
#     voxel_size=np.array([8, 8, 8]),
#     dtype=np.uint8,
#     write_size=Coordinate(np.array([64, 64, 64]) * 128),
# )


# ## validation crop

# ### original smaller validation region

# In[2]:


import pandas as pd
import numpy as np
from funlib.geometry import Roi, Coordinate
from funlib.persistence import prepare_ds

# densely annotated validation region, she wanted the newer one to be twice the size in z
offset = (
    np.array([27400, 2000, 5100])[::-1] * 8
)  # originally had z be 5300, but she wanted it twice the size in z
dimensions = (
    np.array([300, 200, 400])[::-1] * 8
)  # originally had z be 200, but then wanted it twice the size in z
roi = Roi(offset, dimensions)

output_ds = prepare_ds(
    "/nrs/cellmap/ackermand/forGrace/jrc_22ak351-leaf-3m.n5",
    "validation_annotation_box",
    total_roi=roi,
    voxel_size=np.array([8, 8, 8]),
    dtype=np.uint8,
    write_size=Coordinate(np.array([64, 64, 64]) * 128),
)


# ### new validation from grace, generating the box

# In[22]:


import pandas as pd
import numpy as np
from funlib.geometry import Roi, Coordinate
from funlib.persistence import prepare_ds

df = pd.read_csv("annotations_20230802_101047.csv")
all_coords = []
for c in ["z", "y", "x"]:
    all_coords.append(
        np.concatenate(
            (df[f"start {c} (nm)"].to_numpy(), df[f"end {c} (nm)"].to_numpy())
        )
    )
all_coords = np.stack(all_coords)
# pad a bit
mins = all_coords.min(axis=1) - 20 * 8
maxs = all_coords.max(axis=1) + 20 * 8
roi = Roi(mins, maxs - mins).snap_to_grid((8, 8, 8))
print(roi.get_shape() / 8)

output_ds = prepare_ds(
    "/nrs/cellmap/ackermand/forGrace/validation_crop.n5",
    "box",
    total_roi=roi,
    voxel_size=np.array([8, 8, 8]),
    dtype=np.uint8,
    write_size=Coordinate(np.array([64, 64, 64]) * 128),
    # force_exact_write_size=True
)


# new validation from grace, generating the annotations

# In[1]:


from tqdm import tqdm
from numcodecs.gzip import GZip
import zarr
from funlib.geometry import Roi
import pandas as pd
import numpy as np

# annotations_20230829_173628 was generating by resubmitting to get_annotations and hacking it to remove the ones falsesely saved at the wron resolution
df = pd.read_csv("annotations_20230829_173628.csv")
all_coords = []
for c in ["z", "y", "x"]:
    all_coords.append(
        np.concatenate(
            (df[f"start {c} (nm)"].to_numpy(), df[f"end {c} (nm)"].to_numpy())
        )
    )
all_coords = np.stack(all_coords)
# pad a bit
mins = all_coords.min(axis=1) - 20 * 8
maxs = all_coords.max(axis=1) + 20 * 8
roi = Roi(mins, maxs - mins).snap_to_grid((8, 8, 8))
print(roi.begin, roi)
zarr_file = zarr.open(
    f"/nrs/stern/em_data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5", mode="r"
)
dataset = "em/fibsem-uint8/s0"
resolution = np.array(zarr_file[dataset].attrs.asdict()["transform"]["scale"])

pd_starts = (
    np.array([df["start x (nm)"], df["start y (nm)"], df["start z (nm)"]]).T
    / resolution
)
pd_ends = (
    np.array([df["end x (nm)"], df["end y (nm)"], df["end z (nm)"]]).T / resolution
)
pd_centers = list(
    map(tuple, np.round(((pd_starts + pd_ends) * resolution / 2)).astype(int))
)

# get all pd voxels and all overlapping/intersecting voxels between multiple pd
all_pd_voxels_set = set()
intersection_voxels_set = set()
for pd_start, pd_end in tqdm(zip(pd_starts, pd_ends), total=len(pd_starts)):
    voxels_in_cylinder = in_cylinder(pd_start, pd_end, radius=4)
    intersection_voxels_set.update(all_pd_voxels_set.intersection(voxels_in_cylinder))
    all_pd_voxels_set.update(voxels_in_cylinder)


# repeat but now will write out the relevant voxels with appropriate id
store = zarr.N5Store("/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5")
zarr_root = zarr.group(store=store)
ds = zarr_root.create_dataset(
    name="larger_validation_crop",
    dtype="u2",
    shape=zarr_file[dataset].shape,
    chunks=128,
    write_empty_chunks=False,
    compressor=GZip(level=6),
)
attributes = ds.attrs
attributes["pixelResolution"] = {
    "dimensions": 3 * [8],
    "unit": "nm",
}

pd_id = 1
all_pd_voxels_set -= intersection_voxels_set
for pd_start, pd_end, pd_center in tqdm(
    zip(pd_starts, pd_ends, pd_centers), total=len(pd_starts)
):
    if np.all(pd_center[::-1] >= mins) and np.all(pd_center[::-1] <= maxs):
        voxels_in_cylinder = (
            in_cylinder(pd_start, pd_end, radius=4) - intersection_voxels_set
        )
        if len(voxels_in_cylinder) > 0:
            voxels_in_cylinder = np.array(list(voxels_in_cylinder))
            ds[
                voxels_in_cylinder[:, 2],
                voxels_in_cylinder[:, 1],
                voxels_in_cylinder[:, 0],
            ] = pd_id
            pd_id += 1
        else:
            raise Exception(f"Empty plasmodesmata {pd_starts}-{pd_ends}")


# In[17]:


pd_id


# In[53]:


from numcodecs.gzip import GZip

zarr_file = zarr.open(
    f"/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5", mode="r"
)
plasmodesmata_as_cylinders = zarr_file["plasmodesmata_as_cylinders"]
validation_crop = plasmodesmata_as_cylinders[
    offset[2] : offset[2] + dimensions[2],
    offset[1] : offset[1] + dimensions[1],
    offset[0] : offset[0] + dimensions[0],
]

store = zarr.N5Store("/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5")
zarr_root = zarr.group(store=store)
ds = zarr_root.create_dataset(
    overwrite=True,
    name="validation_crop",
    data=validation_crop,
    dtype="u2",
    chunks=128,
    write_empty_chunks=False,
    compressor=GZip(level=6),
)
attributes = ds.attrs
attributes["pixelResolution"] = {
    "dimensions": 3 * [8],
    "unit": "nm",
}
attributes["offset"] = list(offset)


# visualize removed ones

# # Dacapo

# In[4]:


from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    IntensityAugmentConfig,
)
from dacapo.experiments.tasks import AffinitiesTaskConfig
from funlib.geometry.coordinate import Coordinate
import math


# ## Trainer

# In[5]:


trainer_config = GunpowderTrainerConfig(
    name="default_v2_no_dataset_predictor_node_lr_5E-5",
    batch_size=2,
    learning_rate=0.00005,
    augments=[
        ElasticAugmentConfig(
            control_point_spacing=(100, 100, 100),
            control_point_displacement_sigma=(10.0, 10.0, 10.0),
            rotation_interval=(0, math.pi / 2.0),
            subsample=8,
            uniform_3d_rotation=True,
        ),
        IntensityAugmentConfig(
            scale=(0.7, 1.3),
            shift=(-0.2, 0.2),
            clip=True,
        ),
    ],
    clip_raw=True,
    num_data_fetchers=20,
    snapshot_interval=10000,
    min_masked=0.05,
    add_predictor_nodes_to_dataset=False,
)


# ## Task

# In[6]:


task_config = AffinitiesTaskConfig(
    name=f"3d_lsdaffs_weight_ratio_1.00",
    neighborhood=[
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
        (9, 0, 0),
        (0, 9, 0),
        (0, 0, 9),
    ],
    lsds=True,
    lsds_to_affs_weight_ratio=1.00,
)


# ## Architecture

# I had an issue where, by default, I created the rasterization at the same resolution as the raw data. But the default architecture (with the upsampling layer `upsample_factors`) expects it to be at 2x the resolution including mask and validation. This resulted in an error when submitting. Since we don't really care about a higher res (at the moment), we can just comment out the upsampling layer (`constant_upsample` and `upsample_factors`)

# In[7]:


architecture_config = CNNectomeUNetConfig(
    name="unet",
    input_shape=Coordinate(216, 216, 216),
    eval_shape_increase=Coordinate(72, 72, 72),
    fmaps_in=1,
    num_fmaps=12,
    fmaps_out=72,
    fmap_inc_factor=6,
    downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
    # constant_upsample=True,
    # upsample_factors=[(2, 2, 2)],
)


# ## Datasplit

# EVERYTHING MUST BE IN Z,Y,X AND NM!

# In[12]:


# # use centers
# from pathlib import Path
# from dacapo.experiments.datasplits.datasets.arrays import (
#     ZarrArrayConfig,
#     IntensitiesArrayConfig,
#     CropArrayConfig,
# )
# from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
# from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
# from funlib.geometry import Roi

# raw_config = ZarrArrayConfig(
#     name="raw",
#     file_name=Path("/nrs/stern/em_data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5"),
#     dataset="em/fibsem-uint8/s0",
# )
# # We get an error without this, and will suggests having it as such https://cell-map.slack.com/archives/D02KBQ990ER/p1683762491204909
# raw_config = IntensitiesArrayConfig(
#     name="raw", source_array_config=raw_config, min=0, max=255
# )

# gt_config = ZarrArrayConfig(
#     name="plasmodesmata",
#     file_name=Path("/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5"),
#     dataset="plasmodesmata_as_cylinders",
# )

# # mask out regions of overlapping plasmodesmata
# mask_config = ZarrArrayConfig(
#     name="mask",
#     file_name=Path("/nrs/cellmap/ackermand/cellmap/leaf-gall/masks.zarr"),
#     dataset="jrc_22ak351-leaf-3m",
# )

# # could do validation as a file
# # val_gt_config = ZarrArrayConfig(
# #     name="plasmodesmata", file_name="/path/to/data.zarr", dataset="labels_val"
# # )

# # NOTE: Everything has to be in z,y,x
# validation_roi = Roi(offset[::-1] * resolution, dimensions[::-1] * resolution)
# val_gt_config = CropArrayConfig(
#     "val_gt", source_array_config=gt_config, roi=validation_roi
# )
# training_data_config = RawGTDatasetConfig(
#     "train",
#     raw_config=raw_config,
#     gt_config=gt_config,
#     sample_points=[
#         Coordinate(pd_center[::-1]) for pd_center in updated_pd_centers
#     ],  # [Coordinate((29229*8,1862*8,7439*8))], #
#     mask_config=mask_config,
# )
# validation_data_config = RawGTDatasetConfig(
#     "val", raw_config=raw_config, gt_config=val_gt_config, mask_config=mask_config
# )
# datasplit_config = TrainValidateDataSplitConfig(
#     name="plasmodesmata",
#     train_configs=[training_data_config],
#     validate_configs=[validation_data_config],
# )


# In[8]:


# use pseudorandom centers
from pathlib import Path
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    IntensitiesArrayConfig,
    CropArrayConfig,
)
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from funlib.geometry import Roi

raw_config = ZarrArrayConfig(
    name="raw",
    file_name=Path("/nrs/stern/em_data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5"),
    dataset="em/fibsem-uint8/s0",
)
# We get an error without this, and will suggests having it as such https://cell-map.slack.com/archives/D02KBQ990ER/p1683762491204909
raw_config = IntensitiesArrayConfig(
    name="raw", source_array_config=raw_config, min=0, max=255
)

gt_config = ZarrArrayConfig(
    name="plasmodesmata",
    file_name=Path("/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5"),
    dataset="plasmodesmata_as_cylinders",
)

# mask out regions of overlapping plasmodesmata
mask_config = ZarrArrayConfig(
    name="mask",
    file_name=Path("/nrs/cellmap/ackermand/cellmap/leaf-gall/masks.zarr"),
    dataset="jrc_22ak351-leaf-3m",
)

# could do validation as a file
# val_gt_config = ZarrArrayConfig(
#     name="plasmodesmata", file_name="/path/to/data.zarr", dataset="labels_val"
# )

# NOTE: Everything has to be in z,y,x
validation_roi = Roi(
    original_validation_bounding_box.offset[::-1] * preprocess.resolution,
    original_validation_bounding_box.shape[::-1] * preprocess.resolution,
)
val_gt_config = CropArrayConfig(
    "val_gt", source_array_config=gt_config, roi=validation_roi
)
training_data_config = RawGTDatasetConfig(
    f"train",
    raw_config=raw_config,
    gt_config=gt_config,
    sample_points=[
        Coordinate(preprocess.pseudorandom_training_center[::-1])
        for preprocess.pseudorandom_training_center in preprocess.pseudorandom_training_centers
    ],  # [Coordinate((29229*8,1862*8,7439*8))], #
    mask_config=mask_config,
)
validation_data_config = RawGTDatasetConfig(
    "val", raw_config=raw_config, gt_config=val_gt_config, mask_config=mask_config
)
datasplit_config = TrainValidateDataSplitConfig(
    name=f"plasmodesmata_pseudorandom_training_centers_maxshift_18_more_annotations",
    train_configs=[training_data_config],
    validate_configs=[validation_data_config],
)


# ## Run

# In[10]:


from dacapo.experiments import RunConfig
from dacapo.experiments.starts import StartConfig
from dacapo.store.create_store import create_config_store

config_store = create_config_store()

start_config = StartConfig("setup04", "best")
iterations = 200000
validation_interval = 5000
repetitions = 3
for i in range(repetitions):
    run_config = RunConfig(
        name=("_").join(
            [
                "scratch" if start_config is None else "finetuned",
                task_config.name,
                datasplit_config.name,
                architecture_config.name,
                trainer_config.name,
            ]
        )
        + f"__{i}",
        task_config=task_config,
        datasplit_config=datasplit_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        num_iterations=iterations,
        validation_interval=validation_interval,
        repetition=i,
        start_config=start_config,
    )
    config_store.store_run_config(run_config)
# "dacapo run -r {run_config.name}"


# In[11]:


print(run_config.name)


# In[ ]:





# # Prediction Mask

# In[2]:


from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from scipy.ndimage import binary_dilation, distance_transform_edt
import numpy as np

ds = open_ds("/nrs/cellmap/jonesa/jrc_22ak351-leaf-3m/crop352_mask_revised.zarr", "s0")
ds.materialize()
# distance = distance_transform_edt(ds.data > 0)
ds.data = 1 - (ds.data > 0)
ds.data = binary_dilation(ds.data, iterations=3)


translation = Coordinate(np.array([22464.0, 12864.0, 64.0]))

roi = Roi(np.array([22400.0, 12800.0, 0.0]), ds.roi.shape * 128)
print(roi)
output_ds = prepare_ds(
    "/nrs/cellmap/ackermand/cellmap/leaf-gall/prediction_masks.zarr",
    "jrc_22ak351-leaf-3m",
    total_roi=Roi([0, 0, 0], [90408, 51712, 261520]).snap_to_grid((128, 128, 128)),
    voxel_size=np.array([128, 128, 128]),
    dtype=np.uint8,
    write_size=Coordinate(np.array([64, 64, 64]) * 128),
    # force_exact_write_size=True
)
output_ds[roi] = ds.data


# In[2]:


from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from scipy.ndimage import binary_dilation, distance_transform_edt
import numpy as np

for iterations in range(1, 3):
    ds = open_ds(
        "/nrs/cellmap/jonesa/jrc_22ak351-leaf-3m/crop352_mask_revised.zarr", "s0"
    )
    ds.materialize()
    # distance = distance_transform_edt(ds.data > 0)
    ds.data = 1 - (ds.data > 0)
    ds.data = binary_dilation(ds.data, iterations=iterations)

    translation = Coordinate(np.array([22464.0, 12864.0, 64.0]))

    roi = Roi(np.array([22400.0, 12800.0, 0.0]), ds.roi.shape * 128)
    print(roi)
    output_ds = prepare_ds(
        "/nrs/cellmap/ackermand/cellmap/leaf-gall/prediction_masks.zarr",
        f"dilation_iterations_{iterations}_jrc_22ak351-leaf-3m",
        total_roi=Roi([0, 0, 0], [90408, 51712, 261520]).snap_to_grid((128, 128, 128)),
        voxel_size=np.array([128, 128, 128]),
        dtype=np.uint8,
        write_size=Coordinate(np.array([64, 64, 64]) * 128),
        # force_exact_write_size=True
    )
    output_ds[roi] = ds.data


# ### grace extended the block that contains the validation region: (see next few cells)

# In[9]:


from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from scipy.ndimage import binary_dilation, distance_transform_edt
import numpy as np
from tifffile import tifffile

im = tifffile.imread(
    "/groups/cellmap/cellmap/parkg/forDavid/larger_validation_region.tif"
)

im = 1 - (im > 0)
im = binary_dilation(im, iterations=3)

output_ds = prepare_ds(
    "/nrs/cellmap/ackermand/cellmap/leaf-gall/validation_masks.zarr",
    "jrc_22ak351-leaf-3m",
    total_roi=Roi([0, 0, 0], [90408, 51712, 261520]).snap_to_grid((128, 128, 128)),
    voxel_size=np.array([128, 128, 128]),
    dtype=np.uint8,
    write_size=Coordinate(np.array([64, 64, 64]) * 128),
)

validation_roi = Roi((19952, 9736, 153344), (13464, 14064, 15104)).snap_to_grid(
    (128, 128, 128)
)
output_ds[validation_roi] = im


# ## write out raw data to tiff to see if it will work with cellpose

# In[7]:


from funlib.persistence import open_ds
from tifffile import tifffile

raw_data = open_ds(
    "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5", "em/fibsem-uint8/s5"
)
raw_data.materialize()
tifffile.imwrite(
    "/nrs/cellmap/ackermand/cellpose/jrc_22ak351-leaf-3m/raw_s5.tif", raw_data.data
)
tifffile.imwrite(
    "/nrs/cellmap/ackermand/cellpose/jrc_22ak351-leaf-3m/raw_s5_inverted.tif",
    255 - raw_data.data,
)


# ### after running cellpose

# In[22]:


from funlib.persistence import open_ds, prepare_ds
from tifffile import tifffile
import numpy as np

cellpose_results = np.load(
    "/nrs/cellmap/ackermand/cellpose/jrc_22ak351-leaf-3m/raw_s5_seg.npy",
    allow_pickle=True,
)[()]
tifffile.imread()
raw_data = open_ds(
    "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5", "em/fibsem-uint8/s5"
)
ds = prepare_ds(
    filename="/nrs/cellmap/ackermand/cellpose/jrc_22ak351-leaf-3m/cellpose_results.n5",
    ds_name="/raw_s5_cp_masks",
    total_roi=raw_data.roi,
    voxel_size=raw_data.voxel_size,
    write_size=raw_data.voxel_size * 64,
    dtype=np.uint16,
    delete=True,
)
ds[raw_data.roi] = cellpose_results["masks"]


# In[2]:


from funlib.persistence import open_ds, prepare_ds
from tifffile import tifffile
import numpy as np

cellpose_results = tifffile.imread(
    "/nrs/cellmap/rhoadesj/tmp_data/tiffs/jrc_22ak351-leaf-3m/raw_s4_inverted_cp_masks.tif"
)
raw_data = open_ds(
    "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5", "em/fibsem-uint8/s4"
)
ds = prepare_ds(
    filename="/nrs/cellmap/ackermand/cellpose/jrc_22ak351-leaf-3m/cellpose_results.n5",
    ds_name="/raw_s4_inverted_cp_masks_from_jeff",
    total_roi=raw_data.roi,
    voxel_size=raw_data.voxel_size,
    write_size=raw_data.voxel_size * 64,
    dtype=np.uint16,
    delete=True,
)
ds[raw_data.roi] = cellpose_results


# In[ ]:


from funlib.persistence import open_ds, prepare_ds
from tifffile import tifffile
import numpy as np

cellpose_results = tifffile.imread(
    "/nrs/cellmap/rhoadesj/tmp_data/tiffs/jrc_22ak351-leaf-3m/raw_s4_inverted_cp_masks.tif"
)
raw_data = open_ds(
    "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5", "em/fibsem-uint8/s4"
)
ds = prepare_ds(
    filename="/nrs/cellmap/ackermand/cellpose/jrc_22ak351-leaf-3m/cellpose_results.n5",
    ds_name="raw_s4_inverted_cp_masks_from_jeff_inverted",
    total_roi=raw_data.roi,
    voxel_size=raw_data.voxel_size,
    write_size=raw_data.voxel_size * 64,
    dtype=np.uint8,
    delete=True,
)
ds[raw_data.roi] = 1 - (cellpose_results > 0)


# ## same for validation region since it extends a bit beyond the original annotations

# In[17]:


from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from scipy.ndimage import binary_dilation, distance_transform_edt
import numpy as np

current_cell_labels_ds = open_ds(
    "/nrs/cellmap/jonesa/jrc_22ak351-leaf-3m/crop352_mask_revised.zarr", "s0"
)
translation = Coordinate(np.array([22464.0, 12864.0, 64.0]))
roi = Roi(np.array([22400.0, 12800.0, 0.0]), current_cell_labels_ds.roi.shape * 128)
current_cell_labels_ds.roi = roi
current_cell_labels_ds.data_roi = roi
current_cell_labels_ds.voxel_size = Coordinate(128, 128, 128)

validation_roi = Roi((19952, 9736, 153344), (13464, 14064, 15104)).snap_to_grid(
    (128, 128, 128)
)

current_cell_labels = current_cell_labels_ds.to_ndarray(validation_roi, fill_value=0)

raw_ds = open_ds(
    "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5", "em/fibsem-uint8/s4"
)
raw = raw_ds.to_ndarray(validation_roi)


import tifffile

tifffile.imwrite(
    "/nrs/cellmap/ackermand/forGrace/larger_validation_region/cells.tif",
    current_cell_labels,
)
tifffile.imwrite(
    "/nrs/cellmap/ackermand/forGrace/larger_validation_region/raw.tif", raw
)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funlib.persistence import open_ds

mask = open_ds(
    "/nrs/cellmap/jonesa/jrc_22ak351-leaf-3m/crop352_mask_revised.zarr/", "s0"
)
frags = pd.read_csv(
    "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3m.n5/fragments_relabeled.csv"
)
v = frags["Volume (nm^3)"].to_numpy()
print(np.sum(v < 10 * 10 * 10 * 8 * 8 * 8) / len(v), len(v))
plt.hist(v, bins=list(range(0, 3_000_000, 100000)))


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from funlib.persistence import open_ds

mask = open_ds(
    "/nrs/cellmap/jonesa/jrc_22ak351-leaf-3m/crop352_mask_revised.zarr", "s0"
)
frags = pd.read_csv(
    "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3m.n5/fragments_relabeled.csv"
)
volume = frags["Volume (nm^3)"].to_numpy()
offsets = np.array([22400.0, 12800.0, 0])
count = 0
ids = frags["Object ID"].to_numpy()
com_z = frags["COM Z (nm)"].to_numpy() - offsets[0]
com_y = frags["COM Y (nm)"].to_numpy() - offsets[1]
com_x = frags["COM X (nm)"].to_numpy() - offsets[2]
ids_to_remove = np.zeros_like(ids)
for idx, (id, v, z, y, x) in tqdm(enumerate(zip(ids, volume, com_z, com_y, com_x))):
    try:
        if (
            v < (10**3 * 8**3)
            or mask.data[int(z // 128), int(y // 128), int(x // 128)] > 0
        ):
            ids_to_remove[idx] = 1
            # print(np.sum(v<10*10*10*8*8*8)/len(v),len(v))
            # plt.hist(v,bins=list(range(0,3_000_000,100000)))
            # ds.data
    except:
        ids_to_remove[idx] = 1


# In[15]:


ids_to_remove.sum()


# In[10]:


plt.imshow(mask.data[int(z // 128), ...])


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi
import cc3d

mask = open_ds(
    "/nrs/cellmap/jonesa/jrc_22ak351-leaf-3m/crop352_mask_revised.zarr", "s0"
)
mask.materialize()
connected_components = cc3d.connected_components(mask.data, connectivity=6)
roi = Roi(np.array([22400.0, 12800.0, 0.0]), mask.roi.shape * 128)
segmented_mask = prepare_ds(
    "/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5",
    "crop352_mask_revised_segmented",
    dtype="u1",
    total_roi=roi,
    voxel_size=(128, 128, 128),
    write_size=3 * [128 * 128],
)
segmented_mask[roi] = connected_components.astype(np.uint8)


# In[ ]:





# In[6]:


connected_components.max()


# In[15]:


connected_components.dtype


# In[9]:


np.any(ids_to_remove == 0)


# In[4]:


mask.data[int(z // 128), int(y // 128), int(x // 128)]


# In[11]:


mask.data_roi


# In[6]:


len(ids)


# In[7]:


len(ids_to_remove)


# In[15]:


frags["COM X (nm)"].max() // 128


# In[16]:


mask.data.shape


# python scripts/submit.py predict -p configs/cellmap/predictions/plasmodesmata/2023-07-26.yaml -w 100

# In[27]:





# # RESTARTING FAILED RUNS

# the following all failed when i mistakenly rewrote out the plasmodesmata as cylinders file as things were running so they said they couldnt find the files so i am restarting them. they all failed at 90k checkpoints as the last save done

# In[23]:


from dacapo.experiments import RunConfig
from dacapo.experiments.starts import StartConfig
from dacapo.store.create_store import create_config_store

config_store = create_config_store()

iterations = 200000
validation_interval = 5000
so turns out i cant do this via here because it wont let me overwrite runs, but i can manually do it in the database and switch the startconfig to the run name and iteration 90000.iterations
so i went into the database and switched setup04 to the parent runname and changed "best"
 to "90000"
# for run in [
#     "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1",
#     "finetuned_3d_lsdaffs_weight_ratio_0.10_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1",
#     "finetuned_3d_lsdaffs_weight_ratio_0.01_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1",
# ]:
#     start_config = StartConfig(run, "90000")
#     run_config = RunConfig(
#         name=run,
#         task_config=task_config,
#         datasplit_config=datasplit_config,
#         architecture_config=architecture_config,
#         trainer_config=trainer_config,
#         num_iterations=iterations,
#         validation_interval=validation_interval,
#         repetition=1,
#         start_config=start_config,
#     )
#     config_store.store_run_config(run_config)


# Visualize run before train: https://github.com/janelia-cellmap/ml_experiments/blob/main/scripts/visualize_pipeline.py
# creates a neuroglancer view, of your training samples. hitting "t" refreshes to a new example

# python ~/Programming/ml_experiments/scripts/visualize_pipeline.py visualize-pipeline -r finetuned_3d_lsdaffs_plasmodesmata_upsample-unet_default_v2__1

# for now i manually created a file in ~/Programming/ml_experiments/configs/cellmap/runs/pd_test/runs.yaml which contains the run names then i changed to the ml_experiments directory and activate the `cellmap_experiments` dir. then i ran `python scripts/submit.py run -r configs/cellmap/runs/pd_test/runs.yaml -b cellmap -q gpu_t4 -n 10`. 

# ## Important Note

# Had to change to `affs.astype(np.float64)` in line 49 /groups/scicompsoft/home/ackermand/Programming/dacapo/dacapo/experiments/tasks/post_processors/watershed_post_processor.py", line 48, in process
# 
# otherwise got this error:
# TypeError: argument 'affinities': type mismatch:
#  from=float32, to=float64
# 
# also had to add flattening and reshapping during remap in line 69 in same file:
# `segmentation = npi.remap(segmentation.flatten(), filtered_fragments, replace).reshape(segmentation.shape)`
# 
# otherwise got error:
# TypeError: '<' not supported between instances of 'int' and 'bytes'
# 
# Also had to change a lot in `/groups/scicompsoft/home/ackermand/Programming/dacapo/dacapo/validate.py` and `/groups/scicompsoft/home/ackermand/Programming/dacapo/dacapo/evaluate.py` because previously it would determine the best validation trial as being when a parameter setting produced better results than its previous best. This means that even if parameter_id = 0 was bad, as long as it was better than parameter_id=0 from all previous validations, then it would overwrite the best. I changed it so that the overall best for the metric, eg. `voi` is calculated first, and the best validation set is only overwritten if it is the overall best.

# In[27]:


from numcodecs.gzip import GZip
import zarr

store = zarr.N5Store("./test.n5")
zarr_root = zarr.group(store=store)
ds = zarr_root.create_dataset(
    overwrite=True,
    name="test",
    dtype="u1",
    shape=zarr_file[dataset].shape,
    chunks=128,
    write_empty_chunks=False,
    compressor=GZip(level=6),
)
attributes = ds.attrs
attributes["pixelResolution"] = {
    "dimensions": 3 * [8],
    "unit": "nm",
}


# In[25]:


import pickle
import numpy_indexed as npi
import numpy as np
from numpy_indexed.funcs import *
from numpy_indexed.index import *
from builtins import *
from numpy_indexed.arraysetops import indices

with open("remap.pkl", "rb") as f:
    segmentation, filtered_fragments, replace = pickle.load(f)
    input = segmentation
    keys = filtered_fragments
    values = replace
# idx = indices(keys, input.flatten(), missing='mask')
segmentation = npi.remap(segmentation, keys, values).reshape(segmentation.shape)


# ## Prediction

# 1. symlink to default directory `/nrs/cellmap/data/` as seen in `ml_experiments/configs/yamls/constants.yaml`
# 2. add `plasmodesmata` to `ml_experiments/configs/yamls/targets/instance_seg.yaml`
# 3. copy `ml_experiments/configs/cellmap/predictions/jrc_mu-liver-zon-1/2023-02-10.yaml` to `ml_experiments/configs/cellmap/predictions/pd_test/2023-05-17.yaml` and update accordingly.
# 4. Try to predict using this: `python scripts/submit.py predict -p configs/cellmap/predictions/pd_test/2023-05-17.yaml -w 60`, based on this region [this region](http://renderer.int.janelia.org:8080/ng/#!%7B%22dimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22position%22:%5B228016.65625%2C14467.8486328125%2C58427.31640625%5D%2C%22crossSectionOrientation%22:%5B0%2C1%2C0%2C0%5D%2C%22crossSectionScale%22:9.803848825583122%2C%22projectionOrientation%22:%5B-0.33944520354270935%2C-0.9385530948638916%2C-0.019331367686390877%2C-0.05934092774987221%5D%2C%22projectionScale%22:6636.598844056%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22n5://http://renderer.int.janelia.org:8080/n5_sources/stern/jrc_22ak351-leaf-3m.n5/em/fibsem-uint8%22%2C%22tab%22:%22source%22%2C%22shaderControls%22:%7B%22normalized%22:%7B%22range%22:%5B40%2C140%5D%7D%7D%2C%22crossSectionRenderScale%22:0.12964438055577399%2C%22name%22:%22fibsem-uint8%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%5B%22n5://http://cellmap-vm1.int.janelia.org/nrs/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5/plasmodesmata_column_cells/%22%2C%22precomputed://http://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/leaf-gall/jrc_22ak351-leaf-3m/plasmodesmata_column_cells/multires%22%5D%2C%22tab%22:%22segments%22%2C%22selectedAlpha%22:0.43%2C%22meshSilhouetteRendering%22:2.6%2C%22segmentDefaultColor%22:%22#9900ff%22%2C%22name%22:%22plasmodesmata_column_cells%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%5B%22n5://http://cellmap-vm1.int.janelia.org/nrs/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5/plasmodesmata_column_target_cells%22%2C%22precomputed://http://cellmap-vm1.int.janelia.org/nrs/ackermand/meshes/multiresolution/leaf-gall/jrc_22ak351-leaf-3m/plasmodesmata_column_target_cells/multires%22%5D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:0.2%2C%22segmentDefaultColor%22:%22#1fffb4%22%2C%22name%22:%22plasmodesmata_column_target_cells%22%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%7B%22url%22:%22local://annotations%22%2C%22transform%22:%7B%22outputDimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22inputDimensions%22:%7B%220%22:%5B8e-9%2C%22m%22%5D%2C%221%22:%5B8e-9%2C%22m%22%5D%2C%222%22:%5B8e-9%2C%22m%22%5D%7D%7D%7D%2C%22tool%22:%22annotateLine%22%2C%22tab%22:%22source%22%2C%22annotations%22:%5B%5D%2C%22shader%22:%22#uicontrol%20float%20lineWidth%20slider%28min=1%2C%20max=50%2C%20step=1%2C%20default=10%29%5Cn#uicontrol%20vec3%20color%20color%28default=%5C%22white%5C%22%29%5Cn%5Cnvoid%20main%28%29%20%7B%5Cn%20%20setLineWidth%28lineWidth%29%3B%5Cn%20%20setColor%28color%29%3B%5Cn%7D%5Cn%5Cn%22%2C%22shaderControls%22:%7B%22color%22:%22#ff0000%22%7D%2C%22name%22:%22annotations%22%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22precomputed://https://cellmap-vm1.int.janelia.org/dm11/ackermand/neuroglancer_annotations/20230510_114340%22%2C%22tab%22:%22source%22%2C%22annotationColor%22:%22#8b8b23%22%2C%22shader%22:%22#uicontrol%20float%20lineWidth%20slider%28min=1%2C%20max=50%2C%20step=1%2C%20default=10%29%5Cn#uicontrol%20vec3%20color%20color%28default=%5C%22white%5C%22%29%5Cn%5Cnvoid%20main%28%29%20%7B%5Cn%20%20setLineWidth%28lineWidth%29%3B%5Cn%20%20setColor%28color%29%3B%5Cn%7D%5Cn%5Cn%22%2C%22shaderControls%22:%7B%22color%22:%22#ff0000%22%7D%2C%22name%22:%22saved_annotations%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22n5://http://cellmap-vm1.int.janelia.org/nrs/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5/validation_crop/%22%2C%22transform%22:%7B%22matrix%22:%5B%5B1%2C0%2C0%2C219200%5D%2C%5B0%2C1%2C0%2C16000%5D%2C%5B0%2C0%2C1%2C42400%5D%5D%2C%22outputDimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%7D%2C%22subsources%22:%7B%22default%22:true%2C%22bounds%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22name%22:%22validation_crop%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22validation_crop%22%7D%2C%22layout%22:%224panel%22%2C%22statistics%22:%7B%22visible%22:true%7D%2C%22selection%22:%7B%22layers%22:%7B%22annotations%22:%7B%22annotationId%22:%22df5661d17178a2ccd1624582136e59c925677961%22%2C%22annotationSource%22:0%2C%22annotationSubsource%22:%22default%22%7D%7D%7D%7D)
# 
# Once we have lsds we need to do affinities
# 
# 1. `python scripts/post_processing_david/02_extract_fragments_blockwise.py`
# 2. 

# In[17]:


import yaml
from pathlib import Path

prediction_data = yaml.safe_load(
    Path(
        "/groups/scicompsoft/home/ackermand/Programming/ml_experiments/configs/cellmap/predictions/pd_test/2023-05-17.yaml"
    )
    .open("r")
    .read()
)
for matrix in prediction_data["predictions"]:
    offset, shape = matrix.get("roi", (None, None))
    print(offset, shape)


# # scratchspace

# ## watershed

# In[4]:


import mwatershed
import numpy as np

offsets = [(0, 1), (1, 0)]
affinities = (
    np.array(
        # [[[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        # [[0, 0, 0], [1, 1, 1], [0, 0, 0]]],
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        dtype=float,
    )
    - 0.5
)
# 9 nodes. connecting edges:
# 2-3, 5-6, 8-9, 4-7, 5-8, 6-9
# components: [(1,),(2,3),(4,7),(5,6,8,9)]

components = mwatershed.agglom(affinities, offsets)
print(affinities, components)


# In[15]:


import mwatershed

offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
affinities = (
    np.array(
        [
            [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
            [[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
            [[[1, 0], [0, 1]], [[0, 0], [0, 0]]],
        ],
        dtype=float,
    )
    - 0.5
)
# 8 nodes. connecting edges:
# 1-2, 1-3, 1-7, 4-8, 6-8, 7-8
# components: [(1,2,3,7),(4,6,7,8)]

components = mwatershed.agglom(affinities, offsets)
print(components)
# assert set(np.unique(components)) == set([1, 4])


# In[2]:


from dacapo.predict import predict
from dacapo.compute_context import LocalTorch, ComputeContext
from dacapo.experiments import Run, ValidationIterationScores
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)

import torch

from pathlib import Path
import logging

logger = logging.getLogger(__name__)
run_name = "finetuned_3d_lsdaffs_plasmodesmata_upsample-unet_default_v2__0"

config_store = create_config_store()
run_config = config_store.retrieve_run_config(run_name)
run = Run(run_config)

# read in previous training/validation stats

stats_store = create_stats_store()
run.training_stats = stats_store.retrieve_training_stats(run_name)
run.validation_scores.scores = stats_store.retrieve_validation_iteration_scores(
    run_name
)

evaluator = run.task.evaluator
evaluator.set_best(run.validation_scores)

# weights_store = create_weights_store()
# print(weights_store.retrieve_best(run_name, "voi", "val"))

# Initialize the evaluator with the best scores seen so far
# evaluator.set_best(run.validation_scores)

from dacapo.experiments.tasks.evaluators import InstanceEvaluationScores
from dacapo.experiments.datasplits.datasets import Dataset

scores = InstanceEvaluationScores()
for dataset, _, _ in evaluator.best_scores.keys():
    break
overall_best_score = evaluator.get_overall_best(dataset, "voi_merge", scores)
print(overall_best_score)


# In[12]:


z = zarr.open("test.n5", mode="w")
z[1, 2, 3] = 1


# In[33]:


arr = np.zeros([128 * 3] * 3)
starts = np.array([100.5, 20, 5])
ends = np.array([150, 40.5, 20])
radius = 3
index_iterable = np.ndindex(*arr.shape)


# In[58]:


# p = np.indices(arr.shape).reshape((arr.size,3))

in_cylinder(p, a, b, r)


# simplest way to generate the rasterized cylinders would be to open each annotation and writing out annotations in corresponding chunks, then we only have to do ~1900 iterations. and can just write out to appropriate chunks. can also read if the chunk already exists so we can write multiple annotations to same chunk

# In[32]:


np.array(zarr_file[dataset].attrs.asdict()["blockSize"])


# In[53]:


zarr_file[dataset].shape


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/nrs/cellmap/ackermand/cellmap/analysisResults/jrc_mus-liver-zon-1/nucleus.csv"
)
plt.hist(df["Volume (nm^3)"], bins=100)
plt.semilogy()
plt.ylabel("Count")
plt.xlabel("Volume (nm^3)")


# In[16]:


import numpy as np

np.sum(df["Volume (nm^3)"] > 0.25e13)


# In[3]:


import numpy_indexed as npi
import numpy as np

a = np.array([1, 2], dtype=np.uint64)
b = np.array([1, 2], dtype=np.uint64)
c = np.array([1, 2], dtype=np.uint64)

print(npi.remap(a, b, c))
print(a.dtype, b.dtype, c.dtype)


# In[5]:


import numpy as np

adjacent_edge_bias = -1
lr_edge_bias = 1
offsets = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (9, 0, 0),
    (0, 9, 0),
    (0, 0, 9),
]
shift = np.array(
    [adjacent_edge_bias if max(offset) <= 1 else lr_edge_bias for offset in offsets]
).reshape((-1, *((1,) * (len((3, 136, 136, 121)) - 1))))
shift


# In[2]:


from dacapo.store.create_store import (
    create_config_store,
    create_config_store,
    create_weights_store,
)
from dacapo.experiments import Run

import daisy
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi

import click
import numpy as np

import subprocess
import logging
from dacapo.predict import predict
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch
from pathlib import Path
import torch
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.experiments.tasks.post_processors.watershed_post_processor_parameters import (
    WatershedPostProcessorParameters,
)


run_name = "finetuned_3d_lsdaffs_plasmodesmata_upsample-unet_default_v2__0"
config_store = create_config_store()
run_config = config_store.retrieve_run_config(run_name)

config_store = create_config_store()
run_config = config_store.retrieve_run_config(run_name)
run = Run(run_config)

# create weights store and read weights
weights_store = create_weights_store()
weights = weights_store.retrieve_weights(run, 165000)
weights_store._load_best(run, "val/voi")
# run.model.load_state_dict(weights.model)

for validation_dataset in run.datasplit.validate:
    output_roi = validation_dataset.gt.roi
    model = run.model
    raw_array = validation_dataset.raw

    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]
    gt_padding = (output_size - validation_dataset.gt.roi.shape) % output_size
    raise Exception(
        f"Predicting with input size {input_size}, output size {output_size}, gt_padding {gt_padding}"
    )
    # calculate input and output rois

    context = (input_size - output_size) / 2
    if output_roi is None:
        input_roi = raw_array.roi
        output_roi = input_roi.grow(-context, -context)
    else:
        input_roi = output_roi.grow(context, context)
    torch.backends.cudnn.benchmark = True
    run.model.eval()
    prediction_array_identifier = LocalArrayIdentifier(
        Path(
            "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/temp.n5"
        ),
        "pred_original_outputsize",
    )
    predict(
        run.model,
        validation_dataset.raw,
        prediction_array_identifier,
        compute_context=LocalTorch(),
        output_roi=validation_dataset.gt.roi,  # Roi((42400,16000,219200),(108*8,108*8,108*8)),#
    )

    # post_processor = run.task.post_processor
    # post_processor.set_prediction(prediction_array_identifier)

    # output_array_identifier = LocalArrayIdentifier(Path("/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/temp.n5"), "output")

    # prediction_array = ZarrArray.open_from_array_identifier(
    #         prediction_array_identifier
    #     )
    # output_array = ZarrArray.create_from_array_identifier(
    #         output_array_identifier,
    #         [axis for axis in prediction_array.axes if axis != "c"],
    #         prediction_array.roi,
    #         None,
    #         prediction_array.voxel_size,
    #         np.uint64,
    # )
    # post_processed_array = post_processor.process(
    #                  WatershedPostProcessorParameters(id=2, bias=0.5), output_array_identifier
    #             )


# # neuroglancer stuff
# neuroglancer --file /nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5 --dataset predictions/2023-05-24/plasmodesmata_affs_lsds/0__affs --file /groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/temp.n5 --dataset pred_original_outputsize pred_specify_outputsize 

# In[38]:


# for jan
from funlib.geometry import Roi

from dacapo.store.create_store import (
    create_config_store,
    create_config_store,
    create_weights_store,
)
from dacapo.experiments import Run
from dacapo.predict import predict
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch
from pathlib import Path
import torch
from dacapo.experiments.tasks.post_processors.watershed_post_processor_parameters import (
    WatershedPostProcessorParameters,
)


run_name = "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0"
config_store = create_config_store()
run_config = config_store.retrieve_run_config(run_name)

config_store = create_config_store()
run_config = config_store.retrieve_run_config(run_name)
run = Run(run_config)

# create weights store and read weights
weights_store = create_weights_store()
weights = weights_store.retrieve_weights(run, 190000)
weights_store._load_best(run, "val/voi")

validation_dataset = run.datasplit.validate[0]
output_rois = []
# output_rois = [Roi(validation_dataset.gt.roi.begin, 3*[i]) for i in [54*8, 55*8, 108*8, 216*8, 217*8, 324*8]]
# output_rois.append(validation_dataset.gt.roi) # this is not a cube, it is 200x200x300 voxels
# output_rois.append(Roi(validation_dataset.gt.roi.begin, 3*[217*8]))
output_rois.append(
    Roi(validation_dataset.gt.roi.begin, [200 * 8, 200 * 8, 300 * 8])
)  # make it not a cube
# output_rois.append(Roi(validation_dataset.gt.roi.begin, [54*8,54*8,60*8])) # make it not a cube

torch.backends.cudnn.benchmark = True
run.model.eval()
output_path = Path(
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5"
)
for output_roi in output_rois:
    output_dataset = "temp"
    prediction_array_identifier = LocalArrayIdentifier(output_path, output_dataset)
    predict(
        run.model,
        validation_dataset.raw,
        prediction_array_identifier,
        compute_context=LocalTorch(),
        output_roi=output_roi,
    )


# In[36]:


z.create_dataset(
    "predictions/2023-06-17_full/plasmodesmata_affs_lsds/0", shape=(200, 200, 300)
)


# In[3]:


from funlib.geometry import Coordinate, Roi

val_input = validation_dataset.raw.__getitem__(input_roi)

raw_dataset = open_ds(
    "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5", "em/fibsem-uint8/s0"
)
shift = 0
scale = 255
raw_input = (
    raw_dataset.to_ndarray(roi=input_roi, fill_value=shift + scale).astype(np.float32)
    - shift
) / scale
raw_input = np.expand_dims(raw_input, (0, 1))


# In[5]:


model = run.model.to(torch.device("cpu"))
predictions = (
    model.forward(torch.from_numpy(raw_input).float().to(torch.device("cpu")))
    .detach()
    .cpu()
    .numpy()[0],
)


# In[21]:


np.array_equal(raw_input, val_input)


# In[12]:


output_size = run.model.compute_output_shape(run.model.eval_input_shape)[
    1
] * daisy.Coordinate((8, 8, 8))
gt_padding = (output_size - daisy.Coordinate((1600, 1600, 2400))) % output_size


# In[17]:


gt_padding / 8


# In[193]:


from dacapo.store.create_store import create_array_store

array_store = create_array_store()
print(array_store.validation_input_arrays(run.name, validation_dataset.name))


# In[185]:


from funlib.geometry import Roi
import zarr

neighborhood = run_config.task_config.neighborhood
num_channels = run.model.num_out_channels

for aff_or_lsd, n_channels in zip(
    ["affs", "lsds"], [len(neighborhood), num_channels - len(neighborhood)]
):
    out_container = "temp.n5"
    out_dataset = "temp"
    channel = 0
    prepare_ds(
        out_container,
        f"{out_dataset}/{channel}__{aff_or_lsd}",
        total_roi=Roi([0, 0, 0], [128, 128, 128]),
        voxel_size=Coordinate([8, 8, 8]),
        write_size=Coordinate([64, 64, 64]),
        dtype=np.float32,
        num_channels=n_channels,
    )
    root = zarr.open(out_container, mode="a")
    ds = root[f"{out_dataset}/{channel}__{aff_or_lsd}"]
    if out_container.endswith(".zarr"):
        ds.attrs["offsets"] = [n[::-1] for n in neighborhood]
    else:
        ds.attrs["offsets"] = neighborhood


# In[8]:


neighborhood = run_config.task_config.neighborhood
print(type(neighborhood), type(neighborhood[0]))


# In[33]:


import zarr
import matplotlib.pyplot as plt
import numpy as np

out_container = (
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5"
)
root = zarr.open(out_container, mode="r")
ds = root[f"predictions/2023-05-24/plasmodesmata_affs_lsds/0__affs"]
print(ds.shape)

from funlib.persistence import open_ds

print(
    open_ds(
        out_container, "predictions/2023-05-24/plasmodesmata_affs_lsds/0__affs"
    ).shape
)

# plt.imshow(np.mean(ds[:,:,:,0],axis=0),vmax=0.5)
out_container = "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_10_upsample-unet_default_v2__0/validation.zarr"
# root = zarr.open(out_container, mode="r")
# print(root[f"val/voi"].shape)
print(open_ds(out_container, "val/voi").shape)


# In[1]:


from funlib.geometry import Roi
import zarr
import mwatershed as mws
from funlib.segment.arrays import relabel, replace_values
from funlib.persistence import open_ds
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
import time

# densely annotated validation region:
offset = np.array([27400, 2000, 5300])
dimensions = np.array([160, 160, 160])  # [300//4, 200//4, 200//4])
zarr_file = zarr.open(
    f"/nrs/stern/em_data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5", mode="r"
)
dataset = "em/fibsem-uint8/s0"
resolution = np.array(zarr_file[dataset].attrs.asdict()["transform"]["scale"])
validation_roi = Roi(offset[::-1] * resolution, dimensions[::-1] * resolution)

affs = open_ds(
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
    "/predictions/2023-05-24/plasmodesmata_affs_lsds/0__affs",
)
offsets = affs.data.attrs["affs_offsets"]
affs = open_ds(
    "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/temp.n5", "pred"
)
# offsets = [offset[::-1] for offset in offsets]
# offsets=[offsets[i] for i in range(8,-1,-1)]
t = time.time()
offsets = offsets[:]
affs = affs.intersect(validation_roi)
affs.materialize()
affs.data = affs.data[:9].astype(np.float64)
print(affs.data.shape, time.time() - t)
t = time.time()

filter_fragments = 0.5
fragments_data = mws.agglom(
    affs.data - filter_fragments,  # + shift + random_noise + smoothed_affs,
    offsets=offsets,
)
print("agglom", time.time() - t)
t = time.time()
prev_high_mean = 0
if filter_fragments > 0:
    average_affs = np.mean(affs.data, axis=0)

    filtered_fragments = []

    fragment_ids = np.unique(fragments_data)

    for fragment, mean in zip(
        fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
    ):
        if mean < filter_fragments:
            filtered_fragments.append(fragment)
        if mean > filter_fragments:
            # print(fragment,np.sum(fragments_data == fragment))
            prev_high_mean = mean

    filtered_fragments = np.array(filtered_fragments, dtype=fragments_data.dtype)
    replace = np.zeros_like(filtered_fragments)
    replace_values(fragments_data, filtered_fragments, replace, inplace=True)
print("rest", time.time() - t)
t = time.time()
plt.imshow(fragments_data[:, :, 150], interpolation="none")


# In[7]:


offsets[:3]


# In[6]:


from funlib.segment.arrays import replace_values

import zarr
import numpy as np

from pathlib import Path


# from chatgpt
def get_dtype(n):
    if n <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif n <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif n <= np.iinfo(np.uint32).max:
        dtype = np.uint32
    else:
        dtype = np.uint64

    return dtype


container = (
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5"
)
lut = "luts_full/seg_2023-05-24-plasmodesmata_affs_lsds-0_edges_mwatershed.npz"
zarr_container = zarr.open(container)
fragments = zarr_container["processed/2023-05-24/plasmodesmata_affs_lsds/0/fragments"][
    :
]
# fragments_relabeled = zarr_container["processed/2023-05-24/plasmodesmata_affs_lsds/0/fragments_relabeled"]
mapping = np.load(Path(container, lut))["fragment_segment_lut"]

print(len(np.unique(mapping[1,])))

# segments = replace_values(fragments, mapping[0], mapping[1],fragments_relabeled)

# output = "relabeled"
# zarr_container.create_dataset(output, data=segments, overwrite=True)


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funlib.persistence import open_ds

mask = open_ds(
    "/nrs/cellmap/jonesa/jrc_22ak351-leaf-3m/crop352_mask_revised.zarr/", "s0"
)
frags = pd.read_csv(
    "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3m.n5/fragments_relabeled.csv"
)
v = frags["Volume (nm^3)"].to_numpy()
print(np.sum(v < 10 * 10 * 10 * 8 * 8 * 8) / len(v), len(v))
plt.hist(v, bins=list(range(0, 3_000_000, 100000)))


# In[10]:


frags[frags["Volume (nm^3)"] == frags["Volume (nm^3)"].max()]


# In[5]:


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000, 10000, 1, 1, 1])
np.unique(a)


# In[7]:


class temp:
    def __init__(self, a):
        self.test = a


a = temp(4)
a.test
b = a
b.test = 5
print(b.test, a.test)


# In[8]:


a = [1, 2, 3]
a.remove(2)
print(a)


# In[159]:


np.unique(fragments_data)


# In[110]:


plt.imshow(fragments_data[:, :, 120])


# In[79]:


plt.imshow(affs.data[6, :, :, 64])


# In[5]:


import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np

a = np.random.random([50, 50, 50])


@interact(z=(3, 10))
def plot_scroll(z):
    plt.imshow(a[..., z])


# In[6]:


from dacapo.store import create_array_store

array_store = create_array_store()
array_store.validation_input_arrays(run.name, validation_dataset.name)


# In[1]:


import socket
import neuroglancer
import numpy as np

import neuroglancer
import neuroglancer.cli

from funlib.persistence import open_ds


def add_segmentation_layer(state, data, name):
    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units="nm", scales=[8, 8, 8]
    )
    state.dimensions = dimensions
    if name == "raw":
        state.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=data,
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units=["nm", "nm", "nm"],
                    scales=[8, 8, 8],
                    coordinate_arrays=[
                        None,
                        None,
                        None,
                    ],
                ),
                voxel_offset=(0, 0, 0),
            ),
        )
    else:
        state.layers.append(
            name=name,
            segments=[str(i) for i in np.unique(data[data > 0])],
            layer=neuroglancer.LocalVolume(
                data=data,
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units=["nm", "nm", "nm"],
                    scales=[8, 8, 8],
                    coordinate_arrays=[
                        None,
                        None,
                        None,
                    ],
                ),
                voxel_offset=(0, 0, 0),
            ),
        )


f = "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0/validation.zarr"
# f = "/nrs/cellmap/nguyenh3/cellmap/nuclear_pores/ml_results/finetuned_3d_lsdaffs_nuclearpores_upsample-unet_default_v2__test_0/validation.zarr"


voi = open_ds(
    f,
    "val/voi",
).data[:]

voi_merge = open_ds(
    f,
    "val/voi_merge",
).data[:]

voi_split = open_ds(
    f,
    "val/voi_split",
).data[:]
raw = open_ds(
    f,
    "inputs/val/raw",
).data[:]

gt = open_ds(
    f,
    "inputs/val/gt",
).data[:]

processed = open_ds(
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
    "processed/2023-06-17/plasmodesmata_affs_lsds/0/fragments_relabeled",
).data[:]

neuroglancer.set_server_bind_address(
    bind_address=socket.gethostbyname_ex(socket.gethostname())[0]
)
viewer = neuroglancer.Viewer()
with viewer.txn() as state:
    add_segmentation_layer(state, gt, "gt")
    add_segmentation_layer(state, raw, "raw")
    # add_segmentation_layer(state, voi, "best voi")
    # add_segmentation_layer(state, voi_merge, "best voi merge")
    # add_segmentation_layer(state, voi_split, "best voi split")

    # add_segmentation_layer(state, processed, "processed")

print(viewer)


# 

# In[1]:


import socket
import neuroglancer
import numpy as np

import neuroglancer
import neuroglancer.cli

from funlib.persistence import open_ds


def add_segmentation_layer(state, data, name):
    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units="nm", scales=[8, 8, 8]
    )
    state.dimensions = dimensions
    state.layers.append(
        name=name,
        segments=[str(i) for i in np.unique(data[data > 0])],
        layer=neuroglancer.LocalVolume(
            data=data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=[8, 8, 8],
                coordinate_arrays=[
                    None,
                    None,
                    None,
                ],
            ),
            voxel_offset=(0, 0, 0),
        ),
    )


neuroglancer.set_server_bind_address(
    bind_address=socket.gethostbyname(socket.gethostname())
)
viewer = neuroglancer.Viewer()
with viewer.txn() as state:
    for iteration in range(5000, 55000 + 1, 5000):
        val = open_ds(
            "/nrs/cellmap/ackermand/presentations/plasmodesmata/validations.n5",
            f"iteration_{iteration}",
        ).data[:]
        add_segmentation_layer(state, val, f"{iteration}")
print(viewer)


# In[8]:


import socket
import neuroglancer
import numpy as np

import neuroglancer
import neuroglancer.cli

from funlib.persistence import open_ds


def add_segmentation_layer(state, data, name):
    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units="nm", scales=[8, 8, 8]
    )
    state.dimensions = dimensions
    if name == "raw":
        state.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=data,
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units=["nm", "nm", "nm"],
                    scales=[8, 8, 8],
                    coordinate_arrays=[
                        None,
                        None,
                        None,
                    ],
                ),
                voxel_offset=(0, 0, 0),
            ),
        )
    else:
        state.layers.append(
            name=name,
            segments=[str(i) for i in np.unique(data[data > 0])],
            layer=neuroglancer.LocalVolume(
                data=data,
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units=["nm", "nm", "nm"],
                    scales=[8, 8, 8],
                    coordinate_arrays=[
                        None,
                        None,
                        None,
                    ],
                ),
                voxel_offset=(0, 0, 0),
            ),
        )


# f = "/nrs/cellmap/nguyenh3/cellmap/nuclear_pores/ml_results/finetuned_3d_lsdaffs_nuclearpores_upsample-unet_default_v2__test_0/validation.zarr"


voi_original = open_ds(
    "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0/validation.zarr",
    "val/voi",
).data[:]

voi_new = open_ds(
    "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0/validation.zarr",
    "val/voi",
).data[:]

detection_original = open_ds(
    "/nrs/cellmap/ackermand/validation_inference/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0/processed.n5",
    "iteration_85000/WatershedPostProcessorParameters(id=3, bias=0.75)",
).data[:]

detection_new = open_ds(
    # "/nrs/cellmap/ackermand/validation_inference/finetuned_3d_lsdaffs_weight_ratio_0.01_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1/processed.n5",
    # "/iteration_190000/WatershedPostProcessorParameters(id=2, bias=0.5)"
    # "/nrs/cellmap/ackermand/validation_inference/finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1/processed.n5",
    # "iteration_185000/WatershedPostProcessorParameters(id=2, bias=0.5)",
    "/nrs/cellmap/ackermand/validation_inference/finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node_lr_5E-5__0/processed.n5",
    "iteration_180000/WatershedPostProcessorParameters(id=2, bias=0.5)",
).data[:]

removed_dummy = open_ds(
    "/nrs/cellmap/ackermand/validation_inference/finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_removed_dummy_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5__1/processed.n5",
    "iteration_130000/WatershedPostProcessorParameters(id=2, bias=0.5)",
).data[:]

# raw = open_ds(
#     f,
#     "inputs/val/raw",
# ).data[:]

gt = open_ds(
    "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0/validation.zarr",
    "inputs/val/gt",
).data[:]

processed = open_ds(
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
    "processed/2023-06-17/plasmodesmata_affs_lsds/0/fragments_relabeled",
).data[:]

temp_newest = open_ds(
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
    "processed/validation/finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_more_annotations_unet_default_v2_no_dataset_predictor_node_lr_5E-5__0/original_small_validation_box.n5/iteration_200000_filter_val_0.5_lrb_ratio_-0.08_adj_0.5_lr_-1.2_segs",
).data[:]


neuroglancer.set_server_bind_address("0.0.0.0")

viewer = neuroglancer.Viewer()
with viewer.txn() as state:
    add_segmentation_layer(state, gt, "gt")
    add_segmentation_layer(state, voi_original, "old weighting, best voi")
    add_segmentation_layer(state, voi_new, "new weighting, best voi")
    add_segmentation_layer(state, detection_original, "old weighting, best f1")
    add_segmentation_layer(state, detection_new, "new weighting, best f1")
    add_segmentation_layer(state, removed_dummy, "removed_dummy, best f1")
    add_segmentation_layer(state, temp_newest, "temp_newest")
    state.layout = neuroglancer.row_layout(
        [
            neuroglancer.LayerGroupViewer(layers=["gt"], layout="3d"),
            neuroglancer.LayerGroupViewer(
                layers=["old weighting, best voi"], layout="3d"
            ),
            neuroglancer.LayerGroupViewer(
                layers=["new weighting, best voi"], layout="3d"
            ),
            neuroglancer.LayerGroupViewer(
                layers=["old weighting, best f1"], layout="3d"
            ),
            neuroglancer.LayerGroupViewer(
                layers=["new weighting, best f1"], layout="3d"
            ),
            neuroglancer.LayerGroupViewer(
                layers=["removed_dummy, best f1"], layout="3d"
            ),
            neuroglancer.LayerGroupViewer(layers=["temp_newest"], layout="3d"),
        ]
    )
    # add_segmentation_layer(state, raw, "raw")
    # add_segmentation_layer(state, voi, "best voi")
    # add_segmentation_layer(state, voi_merge, "best voi merge")
    # add_segmentation_layer(state, voi_split, "best voi split")

    # add_segmentation_layer(state, processed, "processed")

print(viewer)


# In[54]:


from funlib.persistence import graphs

import daisy

import numpy as np
from funlib.geometry import Roi
from scipy.ndimage import measurements

import logging
import json
import sys
import pymongo
import time
import itertools

from funlib.persistence import open_ds

sample = "2023-05-24/plasmodesmata_affs_lsds/0"
affs = open_ds(
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
    f"predictions/{sample}__affs",
)
fragments = open_ds(
    "/nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
    f"processed/{sample}/fragments",
)
roi = Roi(affs.data_roi.begin, [256 * 8] * 3)
affs = affs.intersect(roi)
fragments = fragments.to_ndarray(affs.roi, fill_value=0)
fragment_ids = np.array([x for x in np.unique(fragments) if x != 0])

num_frags = len(fragment_ids)
frag_mapping = {old: seq for seq, old in zip(range(1, num_frags + 1), fragment_ids)}
rev_frag_mapping = {seq: old for seq, old in zip(range(1, num_frags + 1), fragment_ids)}
for old, seq in frag_mapping.items():
    fragments[fragments == old] = seq
if len(fragment_ids) == 0:
    raise Exception("yo")

print("affs shape: %s", affs.shape)
print("fragments shape: %s", fragments.shape)
# logger.debug("fragments num: %d", n)

# convert affs to float32 ndarray with values between 0 and 1
offsets = affs.data.attrs["affs_offsets"]
affs = affs.to_ndarray()
if affs.dtype == np.uint8:
    affs = affs.astype(np.float32) / 255.0

# COMPUTE EDGE SCORES
# mutex watershed has shown good results when using short range edges
# for merging objects and long range edges for splitting. So we compute
# these scores separately

# separate affinities and offsets by range
adjacents = [offset for offset in offsets if max(offset) <= 1]
lr_offsets = offsets[len(adjacents) :]
affs, lr_affs = affs[: len(adjacents)], affs[len(adjacents) :]

# COMPUTE EDGE SCORES FOR ADJACENT FRAGMENTS
max_offset = [max(axis) for axis in zip(*adjacents)]

# removes the last row/column etc
base_fragments = np.expand_dims(fragments[tuple(slice(0, -m) for m in max_offset)], 0)
base_affs = affs[(slice(None, None),) + tuple(slice(0, -m) for m in max_offset)]

# removes first row/column etc
offset_frags = []
for offset in adjacents:
    offset_frags.append(
        fragments[
            tuple(
                slice(o, (-m + o) if m != o else None)
                for o, m in zip(offset, max_offset)
            )
        ]
    )

offset_frags = np.stack(offset_frags, axis=0)
mask = offset_frags != base_fragments

# cantor pairing function
mismatched_labels = (
    (offset_frags + base_fragments) * (offset_frags + base_fragments + 1) // 2
    + base_fragments
) * mask
mismatched_ids = np.array([x for x in np.unique(mismatched_labels) if x != 0])
adjacent_score = measurements.median(
    base_affs,
    mismatched_labels,
    mismatched_ids,
)
adjacent_map = {
    seq_id: float(med_score)
    for seq_id, med_score in zip(mismatched_ids, adjacent_score)
}

# COMPUTE LONG RANGE EDGE SCORES
max_lr_offset = [max(axis) for axis in zip(*lr_offsets)]
base_lr_fragments = fragments[tuple(slice(0, -m) for m in max_lr_offset)]
base_lr_affs = lr_affs[
    (slice(None, None),) + tuple(slice(0, -m) for m in max_lr_offset)
]
lr_offset_frags = []
for offset in lr_offsets:
    lr_offset_frags.append(
        fragments[
            tuple(
                slice(o, (-m + o) if m != o else None)
                for o, m in zip(offset, max_lr_offset)
            )
        ]
    )
lr_offset_frags = np.stack(lr_offset_frags, axis=0)
lr_mask = lr_offset_frags != base_lr_fragments
# cantor pairing function
lr_mismatched_labels = (
    (lr_offset_frags + base_lr_fragments)
    * (lr_offset_frags + base_lr_fragments + 1)
    // 2
    + base_lr_fragments
) * lr_mask
lr_mismatched_ids = np.array([x for x in np.unique(lr_mismatched_labels) if x != 0])
lr_adjacent_score = measurements.median(
    base_lr_affs,
    lr_mismatched_labels,
    lr_mismatched_ids,
)
lr_adjacent_map = {
    seq_id: float(med_score)
    for seq_id, med_score in zip(lr_mismatched_ids, lr_adjacent_score)
}

for seq_id_u, seq_id_v in itertools.combinations(range(1, num_frags + 1), 2):
    cantor_id_u = ((seq_id_u + seq_id_v) * (seq_id_u + seq_id_v + 1)) // 2 + seq_id_u
    cantor_id_v = ((seq_id_u + seq_id_v) * (seq_id_u + seq_id_v + 1)) // 2 + seq_id_v
    if (
        cantor_id_u in adjacent_map
        or cantor_id_v in adjacent_map
        or cantor_id_u in lr_adjacent_map
        or cantor_id_v in lr_adjacent_map
    ):
        adj_weight_u = adjacent_map.get(cantor_id_u, None)
        adj_weight_v = adjacent_map.get(cantor_id_v, None)
        if adj_weight_u is not None and adj_weight_v is not None:
            adj_weight = (adj_weight_v + adj_weight_u) / 2
            adj_weight += 0.5
        elif adj_weight_u is not None:
            adj_weight = adj_weight_u
            adj_weight += 0.5
        elif adj_weight_v is not None:
            adj_weight = adj_weight_v
            adj_weight += 0.5
        else:
            adj_weight = None
        lr_weight_u = lr_adjacent_map.get(cantor_id_u, None)
        lr_weight_v = lr_adjacent_map.get(cantor_id_v, None)
        if lr_weight_u is None and lr_weight_v is None:
            lr_weight = None
        elif lr_weight_u is None:
            lr_weight = lr_weight_v
        elif lr_weight_v is None:
            lr_weight = lr_weight_u
        else:
            lr_weight = (lr_weight_u + lr_weight_v) / 2
        if rev_frag_mapping[seq_id_u] in [2097255, 2097267] and rev_frag_mapping[
            seq_id_v
        ] in [2097255, 2097267]:
            print(
                rev_frag_mapping[seq_id_u],
                rev_frag_mapping[seq_id_v],
                lr_weight,
                adj_weight,
            )


# In[43]:


cantor_id_u, cantor_id_v


# In[29]:


a = np.random.random((3, 3))
a, a[tuple(slice(1, None) for m in [1, 1])]


# In[4]:


from funlib.segment.arrays import relabel
import numpy as np

relabel(np.array([4, 5, 6, 1000]))


# In[ ]:


neuroglancer --file /nrs/cellmap/ackermand/predictions/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5 --dataset processed/2023-05-24/plasmodesmata_affs_lsds/0/fragments processed/2023-05-24/plasmodesmata_affs_lsds/0/fragments_relabeled --file /nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_upsample-unet_default_v2__0/validation.zarr --dataset inputs/val/gt inputs/val/raw val/voi


# In[2]:


1 / 2 / 0.05


# In[6]:


1 / 2 / 0.95


# ## Check how many voxels are part of affs and lsds

# In[18]:


from funlib.persistence import open_ds
import numpy as np

data = open_ds(
    "/nrs/cellmap/ackermand/presentations/plasmodesmata/predictions.n5",
    "gt",
).data
affs = data[:9, ...]
lsds = data[9:, ...]

data = open_ds(
    "/nrs/cellmap/ackermand/presentations/plasmodesmata/validations.n5",
    "gt",
).data
print(np.sum(affs > 0), np.sum(lsds > 0), np.sum(data[:] > 0))


# In[22]:


print(
    np.sum(affs > 0) / 9,
    np.sum(lsds > 0) * 10 / 10,
    np.sum(data[:] > 0),
    np.sum(lsds == 0) / np.sum(lsds > 0),
)


# In[ ]:





# python scripts/plot.py plot -dir test_plots/pd_weights -r finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0 -r finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1 -r finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0 -r finetuned_3d_lsdaffs_weight_ratio_0.50_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1 -r finetuned_3d_lsdaffs_weight_ratio_0.10_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0 -r finetuned_3d_lsdaffs_weight_ratio_0.10_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1 -r finetuned_3d_lsdaffs_weight_ratio_0.01_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1 -r finetuned_3d_lsdaffs_weight_ratio_0.01_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0 -r finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0 -r finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__1 -r finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node_lr_5E-5__0 -r finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node_lr_5E-5__1 -r finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node_lr_1E-5__0 -r finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node_lr_1E-5__1 -cr voi -pt line

# In[ ]:





# # practice voi

# In[27]:


import numpy as np
from funlib.evaluate import rand_voi, detection_scores
from sklearn.metrics import mutual_info_score
from funlib.segment.arrays import relabel


def my_voi(X, Y):
    n = X.size
    X_i, n_i = np.unique(X, return_counts=True)
    p_i = n_i / n

    Y_j, n_j = np.unique(Y, return_counts=True)
    q_j = n_j / n

    voi = 0
    for i in range(len(X_i)):
        for j in range(len(Y_j)):
            r_ij = np.sum((X == X_i[i]) * (Y == Y_j[j])) / n
            if r_ij > 0:
                voi -= r_ij * (np.log2(r_ij / p_i[i]) + np.log2(r_ij / q_j[j]))

    return voi


# openai:
import numpy as np
from scipy.special import comb


def calculate_entropy(labels):
    try:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        probabilities = label_counts / np.sum(label_counts)
        entropy = -np.sum(probabilities * np.log(probabilities))
    except:
        entropy = 0
    return entropy


def calculate_mutual_information(labels1, labels2):
    n_labels1 = labels1.size
    n_labels2 = labels2.size
    if n_labels1 != n_labels2:
        raise ValueError("Segmentations must have the same number of elements.")

    joint_labels = np.stack((labels1.flatten(), labels2.flatten()), axis=0)
    unique_joint_labels, joint_label_counts = np.unique(
        joint_labels, axis=1, return_counts=True
    )

    probabilities = joint_label_counts / n_labels1
    entropy = -np.sum(probabilities * np.log(probabilities))

    label_counts1 = np.bincount(labels1.flatten().astype(np.int64))
    probabilities1 = label_counts1 / n_labels1
    entropy1 = 0
    for p in probabilities1:
        if p != 0:
            entropy1 = -p * np.log(p)

    label_counts2 = np.bincount(labels2.flatten().astype(np.int64))
    probabilities2 = label_counts2 / n_labels2

    entropy2 = 0
    for p in probabilities2:
        if p != 0:
            entropy2 -= p * np.log(p)

    mutual_information = entropy1 + entropy2 - entropy
    return mutual_information


def calculate_variation_of_information(labels1, labels2):
    # mutual_information = calculate_mutual_information(labels1, labels2)
    mutual_information = mutual_info_score(labels1.flatten(), labels2.flatten())
    entropy1 = calculate_entropy(labels1)
    entropy2 = calculate_entropy(labels2)
    variation_of_information = entropy1 + entropy2 - 2 * mutual_information
    return variation_of_information


def rvoi(X, Y):
    o = rand_voi(X, Y)
    return (o["voi_split"] + o["voi_merge"]) / 2.0


X = np.array(
    [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint64,
)
Y = np.array(
    [
        [30, 30, 0, 0, 4],
        [30, 0, 20, 0, 0],
        [0, 0, 20, 20, 0],
        [0, 0, 0, 20, 0],
        [1, 0, 0, 0, 0],
    ],
    dtype=np.uint64,
)
Z = np.zeros((5, 5), dtype=np.uint64)
relabel(Y, inplace=True)
original_voi = rand_voi(X + 1, Y + 1)
print(
    my_voi(Y, X),
    my_voi(X, Y),
    calculate_variation_of_information(Y, Z),
    calculate_variation_of_information(Z, Y),
    (original_voi["voi_split"] + original_voi["voi_merge"]),
)


# In[9]:


Y


# In[65]:


from funlib.segment.arrays import relabel, replace_values

relabel(Y, inplace=True)
print(Y)


# In[99]:


X


# In[100]:


print(detection_scores(np.ones_like(X), X))
print(detection_scores(X, Y, matching_score="iou", matching_threshold=0.5))


# In[55]:


X, Y


# In[6]:


rand_voi(Y, Z)


# In[27]:


from funlib.persistence import open_ds, prepare_ds

gt = open_ds(
    "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0/validation.zarr",
    "inputs/val/gt",
).data[:]

f = "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0/validation.zarr"
voi_original = open_ds(
    f,
    "val/voi",
).data[:]

voi_new = open_ds(
    "/nrs/cellmap/ackermand/cellmap_experiments/test/finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0/validation.zarr",
    "val/voi",
).data[:]

print("voi")
gt = gt.astype(np.uint64)
for current_voi in [voi_original, voi_new]:
    print(
        my_voi(current_voi, gt),
        my_voi(gt, current_voi),
        calculate_variation_of_information(current_voi, gt),
        calculate_variation_of_information(gt, current_voi),
        rvoi(current_voi, gt),
        rvoi(gt, current_voi),
    )


# In[22]:


0.7149 - 0.6885


# In[19]:


coms = np.random.rand(5, 3)
print(a)
current_com = np.array([1, 2, 3])
print(current_com)
np.linalg.norm(coms - current_com, axis=1)


# In[ ]:


finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0
finetuned_3d_lsdaffs_weight_ratio_1.0_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0


# # Getting best runs

# In[37]:


import json
from dacapo.store.create_store import (
    create_config_store,
)
from dacapo.experiments import Run
import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.DataFrame(
    columns=[
        "run",
        "iteration",
        "parameter",
        "full_path",
        "rand_voi",
        "rand_voi_bkgd",
        "detection_f1",
        "detection_iou_f1",
        "detection_avg_iou",
        "detection_iou_avg_iou",
    ]
)

base_dir = "/nrs/cellmap/ackermand/validation_inference/"
for run_name in os.listdir(base_dir):
    for iteration in range(5000, 200000 + 1, 5000):
        for idx, bias in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
            parameter = f"WatershedPostProcessorParameters(id={idx}, bias={bias})"
            dir_name = f"/nrs/cellmap/ackermand/validation_inference/{run_name}/processed.n5/iteration_{iteration}/{parameter}"
            file_name = f"{dir_name}/attributes.json"
            try:
                with open(file_name) as f:
                    data = json.load(f)
                detection = data["detection"]
                f1 = (
                    2
                    * detection["tp"]
                    / (2 * detection["tp"] + detection["fp"] + detection["fn"])
                )
                detection_iou = data["detection_iou"]
                f1_iou = (
                    2
                    * detection_iou["tp"]
                    / (
                        2 * detection_iou["tp"]
                        + detection_iou["fp"]
                        + detection_iou["fn"]
                    )
                )
                row = [
                    run_name,
                    iteration,
                    parameter,
                    dir_name,
                    data["rand_voi"],
                    data["rand_voi_include_background"],
                    f1,
                    f1_iou,
                    detection["avg_iou"],
                    detection_iou["avg_iou"],
                ]
                df.loc[len(df.index)] = row
            except:
                pass
                # print(run_name,iteration)


# In[38]:


removed_dummy = df[df["run"].str.contains("removed_dummy")]
f1_max = max(
    removed_dummy["detection_f1"].max(), removed_dummy["detection_iou_f1"].max()
)
df_maxs = removed_dummy[
    (removed_dummy["detection_f1"] == f1_max)
    | (removed_dummy["detection_iou_f1"] == f1_max)
]
print(f1_max, df_maxs["full_path"].values)


# In[29]:


f1_max = max(df["detection_f1"].max(), df["detection_iou_f1"].max())
df_maxs = df[(df["detection_f1"] == f1_max) | (df["detection_iou_f1"] == f1_max)]
print(f1_max, df_maxs["full_path"].values)


# In[41]:


iou_max = max(df["detection_avg_iou"].max(), df["detection_iou_avg_iou"].max())
df_maxs = df[
    (df["detection_avg_iou"] == iou_max) | (df["detection_iou_avg_iou"] == iou_max)
]
df_maxs


# In[32]:





# In[28]:


import os


# In[36]:


for run in df["run"].unique():
    for metric in ["rand_voi", "rand_voi_bkgd", "detection_f1"]:
        df_run = df[df["run"] == run]
        df_run.reset_index(inplace=True)
        if "voi" in metric:
            best_idx = df_run[metric].idxmin()
        else:
            best_idx = df_run[metric].idxmax()
        row = df_run.iloc[[best_idx]]
        print(
            f'{row["run"].values[0]} best {metric}: {row[metric].values[0]} at location {row["full_path"].values[0]}'
        )


# In[49]:


maxs = df.groupby(["run", "iteration"])["detection_f1"].max()
for run in df["run"].unique():
    run_maxs = maxs[maxs["run"] == run]
    maxs.plot(x="iteration", y="detection_f1")


# In[54]:


maxs.getgroup(
    "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample"
)


# In[9]:


row


# In[38]:


import json
from dacapo.store.create_store import (
    create_config_store,
)
from dacapo.experiments import Run
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    columns=["run", "iteration", "parameter", "full_path", "rand_voi", "detection_f1"]
)


# config_store = create_config_store()
# run_name = "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0"
# run_config = config_store.retrieve_run_config(run_name)
# run = Run(run_config)
runs = [
    "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__0",
    "finetuned_3d_lsdaffs_plasmodesmata_pseudorandom_training_centers_maxshift_18_upsample-unet_default_v2__1",
    "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__0",
    "finetuned_3d_lsdaffs_weight_ratio_1.00_plasmodesmata_pseudorandom_training_centers_maxshift_18_unet_default_v2_no_dataset_predictor_node__1",
]
for run_name in runs:
    best_voi = {}
    best_voi_include_background = {}
    best_mean_voi = {}
    best_f1 = {}
    for iteration in range(5000, 200000 + 1, 5000):
        best_f1[iteration] = (-1, 1e9)
        best_voi[iteration] = (1e9, -1)
        best_voi_include_background[iteration] = (1e9, -1)
        for idx, bias in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
            filename = (
                f"/nrs/cellmap/ackermand/validation_inference/{run_name}/processed.n5/iteration_{iteration}/WatershedPostProcessorParameters(id={idx}, bias={bias})/attributes.json",
            )

            with open(filename[0]) as f:
                data = json.load(f)
            if data["rand_voi"] < best_voi[iteration][0]:
                best_voi[iteration] = (
                    data["rand_voi"],
                    idx,
                )
            if (
                data["rand_voi_include_background"]
                < best_voi_include_background[iteration][0]
            ):
                best_voi_include_background[iteration] = (
                    data["rand_voi_include_background"],
                    idx,
                )
            detection = data["detection"]
            f1 = (
                2
                * detection["tp"]
                / (2 * detection["tp"] + detection["fp"] + detection["fn"])
            )

            if f1 > best_f1[iteration][0]:
                best_f1[iteration] = (f1, idx)
        i = np.argmax(np.array(list(best_f1.values()))[:, 0])
    print(list(best_f1.keys())[i], list(best_f1.values())[i])
    plt.plot(best_f1.keys(), np.array(list(best_f1.values()))[:, 0])
plt.legend(runs)


# In[10]:


from funlib.persistence import open_ds

ds = open_ds(
    "/nrs/cellmap/nguyenh3/cellmap/predictions/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5",
    "/predictions/2023-07-19-masked/nuclear_pores_affs_lsds/0__affs/",
    "r",
)
ds.roi / 8 / 108


# In[8]:


import pymongo
import time

s = time.time()
mongo_client = pymongo.MongoClient(
    "mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017"
)
db = mongo_client["cellmap_postprocessing_hannah"]
blocks_extracted = db[
    f"2023-07-19-masked/nuclear_pores_affs_lsds/0_fragment_blocks_extracted"
]
document = {
    "block_id": "blah",
    "read_roi": "blah",
    "write_roi": "blah",
    "start": "blah",
    "duration": "blah",
}
blocks_extracted.insert_one(document)
print(list(blocks_extracted.find({"block_id": "blah"})))
print(time.time() - s)


# In[1]:


for i in range(5):
    if i == 2:
        continue
    print(i)


# In[ ]:




