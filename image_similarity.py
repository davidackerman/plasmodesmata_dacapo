# %%
# https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3#:~:text=Image%20similarity%20can%20be%20thought,shape%2C%20texture%2C%20and%20composition.
import torch
import open_clip
from sentence_transformers import util
from PIL import Image

# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms(
#     "ViT-B-16-plus-240", pretrained="laion400m_e32"
# )
model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
)
model.to(device)


def imageEncoder(img):
    img1 = Image.fromarray(img)
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1


def generateScore(test_img, data_img):
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0]) * 100, 2)
    return score


import numpy as np


def vector_to_64bit(vector, num_bits=64, seed=0):
    np.random.seed(seed)  # For reproducibility; remove for randomness
    # Create 64 random hyperplanes
    random_hyperplanes = np.random.randn(num_bits, len(vector))
    # Project the vector onto each hyperplane
    projections = np.dot(random_hyperplanes, vector)
    # Convert projections to binary bits
    binary_hash = (projections > 0).astype(int)

    # Convert binary array to a 64-bit integer
    hash_number = 0
    for bit in binary_hash:
        hash_number = (hash_number << 1) | bit
    return hash_number


# %%

import numpy as np
from scipy.stats import binned_statistic
import dask.array as da
import numpy as np
from funlib.geometry import Roi
import numpy as np
from funlib.geometry import Roi
import scipy
from postprocessing.utils.utils import (
    get_rotation_matrix,
)
from funlib.geometry import Roi
from preprocessing.image_data_interface import ImageDataInterface


def radial_average(volume, axis="z", nbins=100):
    """
    Compute a radial average of a 3D volume in a specified plane.

    Parameters:
      volume : 3D numpy array (shape assumed as (nz, ny, nx))
      axis   : str, one of 'x', 'y', or 'z'
               'z': average in xy plane (rotation about z-axis)
               'y': average in xz plane (rotation about y-axis)
               'x': average in yz plane (rotation about x-axis)
      nbins  : number of radial bins

    Returns:
      result : 2D numpy array where one axis is the coordinate along the rotation axis
               and the other is the radial coordinate.
      bins   : the bin edges used for the radial coordinate.
    """
    nz, ny, nx = volume.shape

    if axis == "z":
        # Rotation around the z-axis: average each xy slice.
        cx, cy = nx // 2, ny // 2
        y, x = np.indices((ny, nx))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).flatten()
        result = np.zeros((nz, nbins))
        bins = np.linspace(0, r.max(), nbins + 1)
        for i in range(nz):
            slice_vals = volume[i].flatten()
            stat, _, _ = binned_statistic(r, slice_vals, statistic="mean", bins=bins)
            result[i, :] = stat
        mirrored_img = np.fliplr(result)

        # Concatenate the original image and its mirror (side by side)
        # Axis 1 corresponds to the width dimension.
        result = np.concatenate((mirrored_img, result), axis=1)
        return result, bins

    elif axis == "y":
        # Rotation around the y-axis: average in xz plane.
        cx, cz = nx // 2, nz // 2
        z, x = np.indices((nz, nx))
        r = np.sqrt((x - cx) ** 2 + (z - cz) ** 2).flatten()
        result = np.zeros((ny, nbins))
        bins = np.linspace(0, r.max(), nbins + 1)
        for i in range(ny):
            slice_vals = volume[:, i, :].flatten()
            stat, _, _ = binned_statistic(r, slice_vals, statistic="mean", bins=bins)
            result[i, :] = stat

        mirrored_img = np.fliplr(result)

        # Concatenate the original image and its mirror (side by side)
        # Axis 1 corresponds to the width dimension.
        result = np.concatenate((mirrored_img, result), axis=1)

        return result, bins

    elif axis == "x":
        # Rotation around the x-axis: average in yz plane.
        cy, cz = ny // 2, nz // 2
        z, y = np.indices((nz, ny))
        r = np.sqrt((y - cy) ** 2 + (z - cz) ** 2).flatten()
        result = np.zeros((nx, nbins))
        bins = np.linspace(0, r.max(), nbins + 1)
        for i in range(nx):
            slice_vals = volume[:, :, i].flatten()
            stat, _, _ = binned_statistic(r, slice_vals, statistic="mean", bins=bins)
            result[i, :] = stat

        # Mirror flip the image across the y-axis (left-right flip)
        mirrored_img = np.fliplr(result)

        # Concatenate the original image and its mirror (side by side)
        # Axis 1 corresponds to the width dimension.
        result = np.concatenate((mirrored_img, result), axis=1)

        return result.astype(np.uint8), bins

    else:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'.")


pad = 180  # want this to be the cube size, but we need to make sure that all rotated cubes contain voxels in this region even when rotated, so we need this larger cubes
extra_pad = int(np.ceil(np.sqrt(3) * pad))


def read_and_align_image(current_pd, idi):
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
    print(roi.begin, roi.end)
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
    return im


# %%
import pandas as pd
import pickle

df = pd.read_csv(
    "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3m/plasmodesmata_cleaned_lines.csv"
)
# with open(
#     f"/groups/scicompsoft/home/ackermand/Programming/plasmodesmata_dacapo/preprocessing/jrc_22ak351-leaf-3r_cylindrical_annotations.pkl",
#     "rb",
# ) as file:
#     data = pickle.load(file)
#     starts = data.annotation_starts * 8
#     ends = data.annotation_ends * 8
#     pds = np.concatenate([starts, ends], axis=1)
# df = pd.DataFrame(
#     pds,
#     columns=[
#         "Line Start Z (nm)",
#         "Line Start Y (nm)",
#         "Line Start X (nm)",
#         "Line End Z (nm)",
#         "Line End Y (nm)",
#         "Line End X (nm)",
#     ],
# )
# %%
import matplotlib.pyplot as plt
from importlib import reload
import preprocessing.image_data_interface

reload(preprocessing.image_data_interface)
from preprocessing.image_data_interface import ImageDataInterface

aligned_idi = ImageDataInterface(
    "/nrs/cellmap/ackermand/cellmap/leaf-gall/aligned_annotations/jrc_22ak351-leaf-3m.zarr/annotations/s0"
)
volume1 = aligned_idi.to_ndarray_ts()
image1 = radial_average(volume1, axis="x", nbins=60)[0]
# image1 = np.repeat(np.repeat(image1, 2, axis=0), 2, axis=1)

# t = imageEncoder(image1)
raw_idi = ImageDataInterface(
    "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s0"
)

# loop over df rows:

for i in range(len(df)):
    row = df.iloc[i]
    # if row["Object ID"] == 44766:
    if i in [173923 - 1, 170826 - 1]:
        current_pd = row[
            [
                "Line Start Z (nm)",
                "Line Start Y (nm)",
                "Line Start X (nm)",
                "Line End Z (nm)",
                "Line End Y (nm)",
                "Line End X (nm)",
            ]
        ].values
    # idi = aligned_idi

    if i == 173923 - 1:
        im = read_and_align_image(current_pd, raw_idi)
        image2 = radial_average(im, axis="x", nbins=180)[0]
        print(f"similarity Score: ", round(generateScore(image1, image2), 2))

        break
    # image2 = np.repeat(np.repeat(image2, 2, axis=0), 2, axis=1)
    if i == 170826 - 1:
        im = read_and_align_image(current_pd, raw_idi)
        image1 = radial_average(im, axis="x", nbins=180)[0]


# %%
# aligned images

from preprocessing.image_data_interface import ImageDataInterface

idi = ImageDataInterface(
    "/nrs/cellmap/ackermand/cellmap/leaf-gall/aligned_annotations/jrc_22ak351-leaf-3mb.zarr/annotations/s0"
)
volume1 = idi.to_ndarray_ts()
image1 = radial_average(volume1, axis="x", nbins=120)[0]

idi = ImageDataInterface(
    "/nrs/cellmap/ackermand/cellmap/leaf-gall/aligned_annotations/jrc_22ak351-leaf-2lb.zarr/annotations/s0"
)
volume2 = idi.to_ndarray_ts()
image2 = radial_average(volume2, axis="x", nbins=120)[0]
# upsample image2
# image2 = np.repeat(np.repeat(image1, 2, axis=0), 2, axis=1)

# plt.imshow(volume1.min(axis=0))

print(f"similarity Score: ", round(generateScore(image1, image2), 2))
# %%
