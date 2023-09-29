import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from numcodecs.gzip import GZip
import os

class PreprocessCylindricalAnnotations:

    def __init__(self, 
                 annotation_csvs,
                 raw_n5="/nrs/stern/em_data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.n5",
                 raw_dataset_name="em/fibsem-uint8/s0",
                 output_n5="/nrs/cellmap/ackermand/cellmap/leaf-gall/jrc_22ak351-leaf-3m.n5",
                 output_dataset_name="plasmodesmata_as_cylinders",
                 mask_zarr="/nrs/cellmap/ackermand/cellmap/leaf-gall/masks.zarr",
                 mask_dataset="jrc_22ak351-leaf-3m"):
        if not isinstance(annotation_csvs, list):
            annotation_csvs = [annotation_csvs]
        self.annotation_csvs = annotation_csvs
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
        resolution = np.array(self.raw_dataset.attrs.asdict()["transform"]["scale"])
        # https://cell-map.slack.com/archives/C04N9JUFQK1/p1683733456153269
        
        df = pd.concat([pd.read_csv(annotation_csv) for annotation_csv in self.annotation_csvs])

        self.pd_starts = (
            np.array([df["start x (nm)"], df["start y (nm)"], df["start z (nm)"]]).T
            / resolution
        )
        self.pd_ends = (
            np.array([df["end x (nm)"], df["end y (nm)"], df["end z (nm)"]]).T / resolution
        )
        self.pd_centers = list(
            map(tuple, np.round(((self.pd_starts + self.pd_ends) * resolution / 2)).astype(int))
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

    def temp(self):

        def point_is_valid_center(pt, edge_length):
            # a point is considerd a valid center if the input bounding box for it does not cross the validation crop
            if np.all((pt + edge_length) >= offset) and np.all(
                (pt - edge_length) <= (offset + dimensions)
            ):
                # then it overlaps validation
                return False
            return True

        def too_close_to_validation(pd_start, pd_end, edge_length):
            # either the start or end will be furthest from the box
            return not (
                point_is_valid_center(pd_start, edge_length)
                or point_is_valid_center(pd_end, edge_length)
            )


        self.pseudorandom_training_centers = []
        removed_ids = []
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
            if not too_close_to_validation(pd_start, pd_end, edge_length):
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
                    tuple(np.round(center * resolution).astype(int))
                )
            else:
                c = np.round(((pd_start + pd_end) * resolution / 2)).astype(int)
                if tuple(c) not in removed_centers:
                    print(pd_start, pd_end)
                    removed_ids.append(id)
        len(pseudorandom_training_centers), len(pd_starts)
        # if self.use_negative_examples:
        #     pseudorandom_training_centers += negative_example_centers
        print(len(pseudorandom_training_centers))