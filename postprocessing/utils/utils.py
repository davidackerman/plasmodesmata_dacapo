import numpy as np
import struct
from funlib.geometry import Coordinate
import tensorstore as ts


def open_ds_tensorstore(dataset_path: str, mode="r"):
    # open with zarr or n5 depending
    filetype = (
        "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
    )
    spec = {
        "driver": filetype,
        "kvstore": {"driver": "file", "path": dataset_path},
    }
    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False)
    else:
        dataset_future = ts.open(spec, read=False, write=True)

    return dataset_future.result()


def to_ndarray_tensorstore(dataset, roi=None, voxel_size=None, offset=None):
    """Read a region of a tensorstore dataset and return it as a numpy array

    Args:
        dataset ('tensorstore.dataset'): Tensorstore dataset
        roi ('funlib.geometry.Roi'): Region of interest to read

    Returns:
        Numpy array of the region
    """
    if roi is None:
        return dataset.read().result()

    if offset is None:
        offset = Coordinate(np.zeros(roi.dims, dtype=int))

    roi -= offset
    roi /= voxel_size

    # Specify the range
    roi_slices = roi.to_slices()

    domain = dataset.domain
    # Compute the valid range
    valid_slices = tuple(
        [
            slice(max(s.start, inclusive_min), min(s.stop, exclusive_max))
            for s, inclusive_min, exclusive_max in zip(
                roi_slices, domain.inclusive_min, domain.exclusive_max
            )
        ]
    )

    # Create an array to hold the requested data, filled with a default value (e.g., zeros)
    output_shape = [s.stop - s.start for s in roi_slices]

    if not dataset.fill_value:
        fill_value = 0
    padded_data = np.ones(output_shape, dtype=dataset.dtype.numpy_dtype) * fill_value
    padded_slices = tuple(
        [
            slice(
                max(inclusive_min - s.start, 0),
                min(exclusive_max, s.stop) + max(inclusive_min - s.start, 0),
            )
            for s, inclusive_min, exclusive_max in zip(
                roi_slices, domain.inclusive_min, domain.exclusive_max
            )
        ]
    )
    # print(padded_slices[0], valid_slices[0], roi_slices[0])
    # Read the region of interest from the dataset
    padded_data[padded_slices] = dataset[valid_slices].read().result()

    return padded_data


def get_rotation_matrix(v1, v2):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    if np.linalg.norm(np.cross(v1, v2)) > 0:
        ssc = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        RU = (
            lambda v1, v2: np.identity(3)
            + ssc(np.cross(v1, v2))
            + np.dot(ssc(np.cross(v1, v2)), ssc(np.cross(v1, v2)))
            * (1 - np.dot(v1, v2))
            / (np.linalg.norm(np.cross(v1, v2)) ** 2)
        )
        rot = np.zeros((4, 4))
        rot[3, 3] = 1

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        rot[:3, :3] = RU(v1, v2)
    else:
        rot = np.identity(4)
    return rot


def extract_precomputed_annotations(path):
    annotation_index = path + "/spatial0/0_0_0"
    with open(annotation_index, mode="rb") as file:
        annotation_index_content = file.read()

    # need to specify which bytes to read
    num_annotations = struct.unpack("<Q", annotation_index_content[:8])[0]
    if (len(annotation_index_content) - 8) % (
        ((6 + 2) * num_annotations * 4)
    ) == 0:  # if it is for a line, there are 6 coordinates to write (4 bytes each), +2 other info stuff?
        annotation_type = "line"
        coords_to_write = 6
    else:
        annotation_type = "point"
        coords_to_write = 3
    annotation_data = struct.unpack(
        f"<Q{coords_to_write*num_annotations}f",
        annotation_index_content[: 8 + coords_to_write * num_annotations * 4],
    )
    annotation_data = np.reshape(
        np.array(annotation_data[1:]), (num_annotations, coords_to_write)
    )

    return annotation_type, annotation_data
