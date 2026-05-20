# %%
import numpy as np
from neuroglancer.write_annotations import AnnotationWriter
from neuroglancer import AnnotationPropertySpec
from neuroglancer.coordinate_space import CoordinateSpace
import numpy as np
import numpy as np
from funlib.geometry import Roi
from funlib.persistence import open_ds
import pandas as pd
from tqdm import tqdm

# %%


# def find_min_max_projected_points(points, line_point, line_direction):
#     # chatgpt
#     line_direction = line_direction / np.linalg.norm(
#         line_direction
#     )  # Normalize direction vector

#     # Calculate the vector from line_point to each point
#     point_vectors = points - line_point

#     # Calculate the projection scalar for each point using dot product and broadcasting
#     projection_scalars = np.sum(point_vectors * line_direction, axis=1)

#     # Calculate the projected points for each point
#     projected_points = line_point + projection_scalars[:, np.newaxis] * line_direction

#     # Find the minimum and maximum projection scalar indices
#     min_projection_idx = np.argmin(projection_scalars)
#     max_projection_idx = np.argmax(projection_scalars)

#     return projected_points[min_projection_idx], projected_points[max_projection_idx]


# # since it is arbitrary to have endpoints for line segment in terms of fitting, will just fit a line and then truncate it
# for ds_name in ["tp_gt", "tp_pred", "fn_gt", "fp_pred"]:
#     annotation_writer = AnnotationWriter(
#         CoordinateSpace(names=("x", "y", "z"), scales=(1, 1, 1), units="nm"),
#         annotation_type="line",
#         properties=[
#             AnnotationPropertySpec(id="identifier", type="uint16"),
#         ],
#     )
#     df = pd.read_csv(
#         f"/nrs/cellmap/ackermand/forAnnotators/leaf-gall/analysisResults/{yaml_name}/{ds_name}.csv"
#     )
#     ds = open_ds(
#         f"/nrs/cellmap/ackermand/forAnnotators/leaf-gall/{yaml_name}.n5", ds_name
#     )
#     # uu, dd, vv = np.linalg.svd(data - datamean, full_matrices=False)
#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         # object id, bounding box and center of mass information, calculated beforehand
#         id = row["Object ID"]
#         cube_min = np.array([row[f"MIN {d} (nm)"] for d in ["Z", "Y", "X"]])
#         cube_max = np.array([row[f"MAX {d} (nm)"] for d in ["Z", "Y", "X"]])
#         com = np.array([row[f"COM {d} (nm)"] for d in ["Z", "Y", "X"]])

#         # define an roi to actually ecompass the bounding box
#         roi = Roi(cube_min - 8, (cube_max - cube_min) + 16)

#         # only look at pixels corresponding to current object
#         data = np.column_stack(np.where(ds.to_ndarray(roi) == id))

#         # fit line to object voxels
#         uu, dd, vv = np.linalg.svd(data - np.mean(data, axis=0), full_matrices=False)
#         line_direction = vv[0]
#         line_origin = com

#         # find endpoints of line segment so that we can write it as neuroglancer annotations
#         start_point, end_point = find_min_max_projected_points(
#             data * 8 + 4 + roi.begin, line_origin, line_direction
#         )

#         # write out lines as neuroglancer annotations
#         annotation_writer.add_line(
#             point_a=start_point[::-1],
#             point_b=end_point[::-1],
#             id=int(id),
#             identifier=int(id),
#         )

#     annotation_writer.write(
#         f"/groups/cellmap/cellmap/ackermand/neuroglancer_annotations/leaf-gall/forAnnotators/{yaml_name}/{ds_name}"
#     )
# %%

import dask.diagnostics
import dask.array as da
from dask.distributed import Client
import numpy as np
from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
from funlib.geometry import Roi
import scipy
from funlib.persistence import open_ds
from utils.utils import get_rotation_matrix, extract_precomputed_annotations
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
from preprocessing.image_data_interface import ImageDataInterface
from preprocessing.zarr_util import create_multiscale_dataset
import pickle

pad = 60  # want this to be the cube size, but we need to make sure that all rotated cubes contain voxels in this region even when rotated, so we need this larger cubes
extra_pad = int(np.ceil(np.sqrt(3) * pad))


@dask.delayed
def read_and_align_image(current_pd, idi: ImageDataInterface):
    resolution = idi.voxel_size[0]
    pd_start = current_pd[:3]
    pd_end = current_pd[3:]
    pd_center = np.round(0.5 * (pd_start + pd_end) / resolution) * resolution
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    v1 = pd_start - pd_end
    # v1 = np.array([1, 0, 0])
    v2 = np.array([0, 0, 1])
    rot = get_rotation_matrix(v1, v2)
    roi = Roi(
        (pd_center - extra_pad * 8),
        [(extra_pad * 2) * 8 + (1 * resolution)] * 3,
    )
    # added 1 above because want it centered

    im = idi.to_ndarray_ts(roi)
    # Translation matrix to shift the image center to the origin
    z, y, x = im.shape
    trans = np.array((-z / 2, -y / 2, -x / 2))

    T = np.identity(4)
    T[:3, -1] = trans
    im = scipy.ndimage.affine_transform(
        im, np.linalg.inv(np.linalg.inv(T).dot(rot).dot(T)), order=0
    )
    scale = int(8 / resolution)
    # extra_pad corresponds to the center, so subtract and add pad to it
    im = im[
        scale * (extra_pad - pad) : scale * (extra_pad + pad) + 1,
        scale * (extra_pad - pad) : scale * (extra_pad + pad) + 1,
        scale * (extra_pad - pad) : scale * (extra_pad + pad) + 1,
    ]
    return da.from_array(im)


# %%
from importlib import reload
import preprocessing.image_data_interface

reload(preprocessing.image_data_interface)
from preprocessing.image_data_interface import ImageDataInterface

suffix_to_raw_dict = {
    "2l": "/nrs/cellmap/data/jrc_22ak351-leaf-2l/jrc_22ak351-leaf-2l.zarr/recon-1/em/fibsem-uint8/s0",
    "3m": "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s0",
    "3r": "/nrs/cellmap/data/jrc_22ak351-leaf-3r/jrc_22ak351-leaf-3r.zarr/recon-1/em/fibsem-uint8/s0",
    "2lb": "/nrs/cellmap/data/jrc_22ak351-leaf-2lb/jrc_22ak351-leaf-2lb.n5/em/fibsem-uint8/s0",
    "3mb": "/nrs/cellmap/data/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb.n5/em/fibsem-uint8/s0",
    "3rb": "/nrs/cellmap/data/jrc_22ak351-leaf-3rb/jrc_22ak351-leaf-3rb.n5/em/fibsem-uint8-v2/s0",
}
# idi = ImageDataInterface(
#     "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s0"
# )
# %%
if __name__ == "__main__":
    # %%
    # Loop over the datasets
    for suffix in ["2l", "3m", "3r"]:  # , "2l", "3m", "3r"]:
        dataset = f"jrc_22ak351-leaf-{suffix}"
        idi = ImageDataInterface(
            suffix_to_raw_dict[suffix],
        )

        with open(
            f"/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/{dataset}_cylindrical_annotations.pkl",
            "rb",
        ) as file:
            data = pickle.load(file)
            starts = data.annotation_starts * 8
            ends = data.annotation_ends * 8
            pds = np.concatenate([starts, ends], axis=1)
        with Client(threads_per_worker=1, n_workers=1) as client:
            client.cluster.scale(10)
            print(client.dashboard_link.replace("127.0.0.1", "ackermand-ws2.hhmi.org"))

            # Extract neuroglancer annotations as numpy array
            # _, pds = extract_precomputed_annotations(
            #     f"/groups/cellmap/cellmap/ackermand/neuroglancer_annotations/leaf-gall/forAnnotators/{yaml_name}/{ds_name}"
            # )
            output_dim = (extra_pad * 2 + 1) * 8 / idi.voxel_size[0]
            # Align the images
            aligned_images = [
                da.from_delayed(
                    read_and_align_image(current_pd, idi),
                    (output_dim, output_dim, output_dim),
                    np.float64,
                )
                for current_pd in pds
            ]
            # # Stack all small Dask arrays into one
            stack = da.stack(aligned_images, axis=0)
            composite_image = stack.mean(axis=0).compute()

            roi = Roi((0, 0, 0), np.array(composite_image.shape) * idi.voxel_size[0])

            output_ds = create_multiscale_dataset(
                f"/nrs/cellmap/ackermand/cellmap/leaf-gall/aligned_annotations/{dataset}.zarr/annotations",
                dtype=np.uint8,
                voxel_size=idi.voxel_size,
                total_roi=roi,
                write_size=Coordinate(3 * [8 * 128]),
            )
            output_ds[roi] = composite_image
        print("done")

# %%
