import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from numcodecs.gzip import GZip
import os

import socket
import neuroglancer
import numpy as np

import neuroglancer
import neuroglancer.cli
import struct
import os
import struct
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import shutil
class PreprocessCylindricalAnnotations:

    def __init__(self, 
                 annotation_csvs,
                 validation_or_test_rois,
                 raw_n5="/nrs/stern/em_data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
                 raw_dataset_name="em/fibsem-uint8/s0",
                 output_n5="/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5",
                 output_dataset_name="plasmodesmata_as_cylinders",
                 mask_zarr="/nrs/cellmap/ackermand/cellmap/leaf-gall/masks.zarr",
                 mask_dataset="jrc_22ak351-leaf-3m"):
        if not isinstance(annotation_csvs, list):
            annotation_csvs = [annotation_csvs]
        self.annotation_csvs = annotation_csvs
        self.validation_or_test_rois = validation_or_test_rois
        zarr_file = zarr.open(raw_n5, mode="r")
        self.raw_dataset = zarr_file[raw_dataset_name]
        self.output_n5 = output_n5
        self.output_dataset_name = output_dataset_name
        self.mask_zarr = mask_zarr
        self.mask_dataset = mask_dataset

    def in_cylinder(self, end_1, end_2, radius):
        # https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
        # normalized tangent vector
        d = np.divide(end_2 - end_1, np.linalg.norm(end_2 - end_1))

        # possible points
        mins = np.floor(np.minimum(end_1, end_2)).astype(int) - (
            np.ceil(radius).astype(int) + 1
        )  # 1s for padding
        maxs = np.ceil(np.maximum(end_1, end_2)).astype(int) + (
            np.ceil(radius).astype(int) + 1
        )

        x, y, z = [list(range(mins[i], maxs[i] + 1, 1)) for i in range(3)]
        p = np.array(np.meshgrid(x, y, z)).T.reshape((-1, 3))

        # signed parallel distance components
        s = np.dot(end_1 - p, d)
        t = np.dot(p - end_2, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros_like(s)])

        # perpendicular distance component
        c = np.linalg.norm(np.cross(p - end_1, d), axis=1)

        is_in_cylinder = (h == 0) & (c <= radius)
        return set(map(tuple, p[is_in_cylinder]))

    def extract_annotation_information(self):
        self.resolution = np.array(self.raw_dataset.attrs.asdict()["transform"]["scale"])
        # https://cell-map.slack.com/archives/C04N9JUFQK1/p1683733456153269
        
        df = pd.concat([pd.read_csv(annotation_csv) for annotation_csv in self.annotation_csvs])

        self.pd_starts = (
            np.array([df["start x (nm)"], df["start y (nm)"], df["start z (nm)"]]).T
            / self.resolution
        )
        self.pd_ends = (
            np.array([df["end x (nm)"], df["end y (nm)"], df["end z (nm)"]]).T / self.resolution
        )
        self.pd_centers = list(
            map(tuple, np.round(((self.pd_starts + self.pd_ends) * self.resolution / 2)).astype(int))
        )

    def get_negative_examples(filename="annotations_20230620_221638.csv"):
        negative_examples = pd.read_csv(filename)
        negative_example_centers = np.array(
            [
                negative_examples["x (nm)"],
                negative_examples["y (nm)"],
                negative_examples["z (nm)"],
            ]
        ).T
        negative_example_centers = list(
            map(tuple, np.round(negative_example_centers).astype(int))
        )
        # ensure no duplicate negative examples
        print(
            len(
                set(
                    (
                        zip(
                            list(negative_examples["x (nm)"]),
                            list(negative_examples["y (nm)"]),
                            list(negative_examples["z (nm)"]),
                        )
                    )
                )
            )
        )
        return negative_example_centers

    def write_annotations_as_cylinders_and_get_intersections(self):
        # get all pd voxels and all overlapping/intersecting voxels between multiple pd
        all_pd_voxels_set = set()
        self.intersection_voxels_set = set()
        for pd_start, pd_end in tqdm(zip(self.pd_starts, self.pd_ends), total=len(self.pd_starts)):
            voxels_in_cylinder = self.in_cylinder(pd_start, pd_end, radius=4)
            self.intersection_voxels_set.update(all_pd_voxels_set.intersection(voxels_in_cylinder))
            all_pd_voxels_set.update(voxels_in_cylinder)

        # repeat but now will write out the relevant voxels with appropriate id
        store = zarr.N5Store(self.output_n5)
        zarr_root = zarr.group(store=store)
        ds = zarr_root.create_dataset(
            name=self.output_dataset_name,
            dtype="u2",  # need to do this, right now it is doing it as floats
            shape=self.raw_dataset.shape,
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
        all_pd_voxels_set -= self.intersection_voxels_set
        for pd_start, pd_end in tqdm(zip(self.pd_starts, self.pd_ends), total=len(self.pd_starts)):
            voxels_in_cylinder = (
                self.in_cylinder(pd_start, pd_end, radius=4) - self.intersection_voxels_set
            )
            if len(voxels_in_cylinder) > 0:
                voxels_in_cylinder = np.array(list(voxels_in_cylinder))
                ds[
                    voxels_in_cylinder[:, 2], voxels_in_cylinder[:, 1], voxels_in_cylinder[:, 0]
                ] = pd_id
                pd_id += 1
            else:
                raise Exception(f"Empty plasmodesmata {self.pd_starts}-{self.pd_ends}")
            
    def write_intersection_mask(self):
        zarr_path = self.mask_zarr
        if not os.path.exists(zarr_path):
            zarr_root = zarr.open(zarr_path, mode="w")
        else:
            zarr_root = zarr.open(zarr_path, mode="r+")
        ds = zarr_root.create_dataset(
            overwrite=True,
            name=self.mask_dataset,
            dtype="u1",
            fill_value=1,
            shape=self.raw_dataset.shape,
            chunks=128,
            write_empty_chunks=False,
            compressor=GZip(level=6),
        )
        attributes = ds.attrs
        attributes["pixelResolution"] = {
            "dimensions": 3 * [8],
            "unit": "nm",
        }
        intersection_voxels = np.array(list(self.intersection_voxels_set))
        ds[intersection_voxels[:, 2], intersection_voxels[:, 1], intersection_voxels[:, 0]] = 0

    def remove_validation_or_test_annotations_from_training(self):

        def point_is_valid_center_for_current_roi(pt, edge_length, offset, shape):
             # a point is considered a valid center if the input bounding box for it does not cross the validation crop
            if np.all((pt + edge_length) >= offset) and np.all(
                (pt - edge_length) <= (offset + shape)
            ):
                # then it overlaps validation
                return False
            return True
        
        def point_is_valid_center(pt, edge_length):
            for roi in self.validation_or_test_rois:
                if not point_is_valid_center_for_current_roi(pt, edge_length, roi.offset, roi.shape):
                    return False
            return True
        
        def too_close_to_rois(pd_start, pd_end, edge_length):
            # either the start or end will be furthest from the box
            return not (
                point_is_valid_center(pd_start, edge_length)
                or point_is_valid_center(pd_end, edge_length)
            )
            
        self.pseudorandom_training_centers = []
        self.removed_ids = []
        for id, pd_start, pd_end in tqdm(
            zip(list(range(1, len(self.pd_starts) + 1)), self.pd_starts, self.pd_ends), total=len(self.pd_starts)
        ):
            # ultimately seems to predict on 36x36x36 region, so we need to make sure this doesn't overlap with validation
            # lets just shift by at most +/-10 in any dimension for the center to help ensure that a non-neglible part of the rasterization, and original annotation, are included in a box centered at that region
            max_shift = 18
            # first find a random coordinate along the annotation. this will be included within the box

            # now find a valid center
            # NB: since we want to make sure that we are far enough away from the validation to ensure that no validation voxels affect training voxels
            # we must make sure the distance is at least the run.model.eval_input_shape/2 = 288/2 = 144
            edge_length = 144 + 1  # add one for padding since we round later on
            if not too_close_to_rois(pd_start, pd_end, edge_length):
                random_coordinate_along_annotation = (
                    pd_start + (pd_end - pd_start) * np.random.rand()
                )
                center = random_coordinate_along_annotation + np.random.randint(
                    low=-max_shift, high=max_shift, size=3
                )
                while not point_is_valid_center(center, edge_length):
                    random_coordinate_along_annotation = (
                        pd_start + (pd_end - pd_start) * np.random.rand()
                    )
                    center = random_coordinate_along_annotation + np.random.randint(
                        low=-max_shift, high=max_shift, size=3
                    )
                self.pseudorandom_training_centers.append(
                    tuple(np.round(center * self.resolution).astype(int))
                )
            else:
                #c = np.round(((pd_start + pd_end) * self.resolution / 2)).astype(int)
                self.removed_ids.append(id)
        print(f"number of original centers: {len(self.pd_starts)}, number of training centers: {len(self.pseudorandom_training_centers)}")
        # if self.use_negative_examples:
        #     pseudorandom_training_centers += negative_example_centers

    def write_out_removed_annotations(self):
        annotation_type = "line"
        output_directory = "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/preprocessing/removed_annotations"
        if os.path.isdir(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(f"{output_directory}/spatial0")
        if annotation_type == "line":
            coords_to_write = 6
        else:
            coords_to_write = 3

        annotations = np.column_stack((self.pd_starts, self.pd_ends))*self.resolution[0]
        annotations = np.array([annotations[id-1,:] for id in self.removed_ids])
        with open(f"{output_directory}/spatial0/0_0_0", "wb") as outfile:
            total_count = len(annotations)
            buf = struct.pack("<Q", total_count)
            for annotation in tqdm(annotations):
                annotation_buf = struct.pack(f"<{coords_to_write}f", *annotation)
                buf += annotation_buf
            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                f"<{total_count}Q", *range(1, len(annotations) + 1, 1)
            )  # so start at 1
            # id_buf = struct.pack('<%sQ' % len(coordinates), 3,1 )#s*range(len(coordinates)))
            buf += id_buf
            outfile.write(buf)

        max_extents = annotations.reshape((-1, 3)).max(axis=0) + 1
        max_extents = [int(max_extent) for max_extent in max_extents]
        info = {
            "@type": "neuroglancer_annotations_v1",
            "dimensions": {"x": [1, "nm"], "y": [1, "nm"], "z": [1, "nm"]},
            "by_id": {"key": "by_id"},
            "lower_bound": [0, 0, 0],
            "upper_bound": max_extents,
            "annotation_type": annotation_type,
            "properties": [],
            "relationships": [],
            "spatial": [
                {
                    "chunk_size": max_extents,
                    "grid_shape": [1, 1, 1],
                    "key": "spatial0",
                    "limit": 1,
                }
            ],
        }

        with open(f"{output_directory}/info", "w") as info_file:
            json.dump(info, info_file)

        print(output_directory.replace(
            "/groups/cellmap/cellmap/ackermand/",
            "precomputed://https://cellmap-vm1.int.janelia.org/dm11/ackermand/",
        ))

    def visualize_removed_annotations(self, roi):
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

        # get data
        expand_by = 500
        expanded_offset = np.array(roi.offset) - expand_by
        expanded_dimension = np.array(roi.shape) + 2 * expand_by
        ds = np.zeros(expanded_dimension, dtype=np.uint64)
        for id, pd_start, pd_end, pd_center in tqdm(
            zip(list(range(1, len(self.pd_starts) + 1)), self.pd_starts, self.pd_ends, self.pd_centers),
            total=len(self.pd_starts),
        ):
            if id in self.removed_ids:
                voxels_in_cylinder = np.array(list(self.in_cylinder(pd_start, pd_end, radius=4)))
                if np.any(np.all(voxels_in_cylinder >= expanded_offset, axis=1) 
                          & np.all(voxels_in_cylinder <= expanded_offset + expanded_dimension, axis=1)):
                    ds[
                        voxels_in_cylinder[:, 2] - expanded_offset[2],
                        voxels_in_cylinder[:, 1] - expanded_offset[1],
                        voxels_in_cylinder[:, 0] - expanded_offset[0],
                    ] = id


        neuroglancer.set_server_bind_address(
            bind_address=socket.gethostbyname(socket.gethostname())
        )
        viewer = neuroglancer.Viewer()
        with viewer.txn() as state:
            add_segmentation_layer(state, ds, "removed")
        print(viewer)
        input("Press Enter to continue...")