# %%
import numpy as np
import pickle
import os
import tifffile
import fastremap
import networkx as nx
from pathlib import Path
import itertools
from tqdm import tqdm
import cc3d

# def get_touching_ids(file_path, output_dir):
#     tifffile.imread(file_path)
#     return None, None


# def graph_recolor_segmentation(file_path, output_dir):

#     im, set_of_touching_ids = get_touching_ids(file_path, output_dir)
#     file_name = Path(file_path).stem
#     G = nx.Graph()
#     G.add_nodes_from(list(range(1, im.max() + 1)))
#     G.add_edges_from(set_of_touching_ids)
#     coloring = nx.coloring.equitable_color(G, num_colors=255)
#     old_values = np.array(list(coloring.keys()), dtype=np.uint16)
#     new_values = np.array(list(coloring.values()), dtype=np.uint16) + 1
#     output_im = im.copy()
#     if output_im.dtype == np.uint8:
#         output_im = output_im.astype(np.uint16)
#     fastremap.remap(
#         output_im,
#         dict(zip(old_values, new_values)),
#         preserve_missing_labels=True,
#         in_place=True,
#     )
#     output_im = output_im.astype(np.uint8)
#     os.makedirs(output_dir, exist_ok=True)
#     tifffile.imwrite(
#         f"{output_dir}/{file_name}_graph_relabeled.tif",
#         output_im,
#     )
#     print(f"Done! Wrote file to {output_dir}/{file_name}_graph_relabeled.tif!")

tifs = [
    "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-2l/crop350/jrc_22ak351-leaf-2l_crop350_whole_cell001_s4-files/Final-Draft-lbl.tif",
    "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-3m/crop352/jrc_22ak351-leaf-3m_crop352_whole_cell001_s4.tif",
    "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-3r/crop361/crop361_labels_convert-lbl.tif",
]
for tif in tifs:
    print(tif)
    file_name = Path(tif).stem
    output_dir = Path(tif).parent
    cells = tifffile.imread(tif)
    if len(fastremap.unique(cells)) == 2:
        cells == 60
    else:
        cells = cells > 0
    cc = cc3d.connected_components(cells, binary_image=True, connectivity=6)
    uniques = fastremap.unique(cc[cc > 0])
    reassigned_uniques = [((unique - 1) % 255) + 1 for unique in uniques]

    remap_dict = dict(zip(uniques, reassigned_uniques))
    fastremap.remap(
        cc,
        remap_dict,
        in_place=True,
        preserve_missing_labels=True,
    )
    cc = cc.astype(np.uint8)
    # write out
    tifffile.imwrite(f"{output_dir}/{file_name}_relabeled.tif", cc)

# %%
# aubrey renamed the output files and i will now save them as zarr
from postprocessing.zarr_util import create_multiscale_dataset
from funlib.geometry import Roi
from tifffile import tifffile
from pathlib import Path
import numpy as np
import fastremap

tifs = [
    "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-2l/crop350/crop350_relabel.tif",
    "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-3m/crop352/crop352_relabeled.tif",
    "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-3r/crop361/crop361_relabeled.tif",
]
# for tif in tifs:
#     file_name = Path(tif).stem
#     if file_name.endswith("_relabel"):
#         file_name += "ed"
#     output_dir = Path(tif).parent
#     im = tifffile.imread(tif)
#     roi = Roi((0, 0, 0), np.array(im.shape) * 128)
#     ds = create_multiscale_dataset(
#         output_path=f"{output_dir}/relabeled.zarr/{file_name}",
#         dtype=np.uint8,
#         voxel_size=3 * [128],
#         total_roi=roi,
#         write_size=3 * [64 * 128],
#     )
#     ds[roi] = im

# need to relabel for connected components for analysis
import cc3d

for tif in tifs:
    dataset = tif.split("/groups/cellmap/cellmap/annotations/amira/")[1].split("/")[0]
    file_name = Path(tif).stem
    if file_name.endswith("_relabel"):
        file_name += "ed"
    output_dir = Path(tif).parent
    im = tifffile.imread(tif)
    im_connected = cc3d.connected_components(im, connectivity=6, binary_image=True)
    im_connected = fastremap.refit(
        im_connected
    )  # resize the data type of the array to fit extrema

    roi = Roi((0, 0, 0), np.array(im.shape) * 128)
    ds = create_multiscale_dataset(
        output_path=f"/nrs/cellmap/ackermand/cellmap/leaf-gall/{dataset}.zarr/cell",
        dtype=im_connected.dtype,
        voxel_size=3 * [128],
        total_roi=roi,
        write_size=3 * [64 * 128],
    )
    ds[roi] = im_connected


# %%
