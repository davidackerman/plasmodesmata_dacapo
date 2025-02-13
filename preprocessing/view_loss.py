# %%
from dacapo.store.create_store import create_stats_store
import itertools
import matplotlib.pyplot as plt
import numpy as np

base_lrs = [2.5e-5, 5e-5]
batch_sizes = [2, 4, 6]
store = create_stats_store()
for bs, base_lr, it in itertools.product(batch_sizes, base_lrs, [0, 1]):
    # create random color
    if it == 0:
        random_color = list(np.random.rand(3))
    lr = base_lr * bs
    if lr <= 0.00005:
        lr = "{:.5f}".format(lr)
    else:
        lr = "{:.4}".format(lr)
    # lr as string without scientific notation
    run_name = f"finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_{lr}_bs_{bs}__{it}"
    training_stats = store.retrieve_training_stats(run_name)

    training_stats.iteration_stats
    xarr = training_stats.to_xarray()
    plt.plot(
        xarr.rolling(iterations=5000).std(),
        label=f"Batch size: {bs}, LR: {lr}",
        linestyle="-" if it == 0 else ":",
        color=random_color,
    )
    plt.ylim(0, 0.2)
plt.legend()
plt.show()
# %%
import torch
from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
from dacapo.store.create_store import create_config_store, create_weights_store

config_store = create_config_store()
weights_store = create_weights_store()
# NOTE: Normally we would just load in run but here we have to recreate it to save time since our run has so many points
from funlib.geometry import Coordinate
from dacapo.experiments.tasks import AffinitiesTaskConfig, AffinitiesTask, DistanceTask
from dacapo.experiments.architectures import CNNectomeUNetConfig, CNNectomeUNet

from dacapo.experiments.tasks import DistanceTaskConfig

task_config = AffinitiesTaskConfig(
    name=f"tmp",
    neighborhood=[
        Coordinate(1, 0, 0),
        Coordinate(0, 1, 0),
        Coordinate(0, 0, 1),
        Coordinate(3, 0, 0),
        Coordinate(0, 3, 0),
        Coordinate(0, 0, 3),
        Coordinate(9, 0, 0),
        Coordinate(0, 9, 0),
        Coordinate(0, 0, 9),
    ],
    lsds=True,
    lsds_to_affs_weight_ratio=0.5,
)

architecture_config = CNNectomeUNetConfig(
    name="unet",
    input_shape=Coordinate(216, 216, 216),
    eval_shape_increase=Coordinate(72, 72, 72),
    fmaps_in=1,
    num_fmaps=12,
    fmaps_out=72,
    fmap_inc_factor=6,
    downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
)
task = AffinitiesTask(task_config)
architecture = CNNectomeUNet(architecture_config)
model = task.create_model(architecture)
weights = weights_store.retrieve_weights(
    "finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00015_bs_6__0",
    65000,
)
model.load_state_dict(weights.model)
model.to("cuda")
model.eval()
# %%
import gc

torch.cuda.empty_cache()
gc.collect()  # Ensure garbage collection happens
ds = open_ds(
    "/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr",
    "recon-1/em/fibsem-uint8/s0",
)
roi = Roi((np.array((30805, 36911, 82318)) // 8) * 8, [8 * (216 + 36 * 4)] * 3)
data = ds.to_ndarray(roi) / 255.0
# prepend batch and channel dimensions
data = data[np.newaxis, np.newaxis, ...].astype(np.float32)
# move to cuda
data = torch.from_numpy(data).to("cuda")
# x = torch.rand((1, 1, 216, 216, 216)).to("cuda")
with torch.no_grad():
    outputs = model(data)
    outputs = outputs.cpu()

import matplotlib.pyplot as plt

plt.imshow(outputs[0, 0, :, :, 0])
del outputs
del data

torch.cuda.empty_cache()
gc.collect()  # Ensure garbage collection happens


# %%
del model
torch.cuda.empty_cache()
gc.collect()  # Ensure garbage collection happens
# %%
import torch
import gc

for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass
# %%
