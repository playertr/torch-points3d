import os, sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

import torch
from torch_points3d.models.grasp_classification.minkowski_graspnet import Minkowski_Baseline_Model
# from torch_points3d.models.grasp_classification.minkowski import Minkowski_Baseline_Model

yaml_config = """
model_name: STRes16UNet14B
data:
    task: grasp_classification
    class: acronymvid.AcronymVidDataset
    name: acronymvid
    dataroot: /home/tim/Research/GraspRefinement/data
    process_workers: 8
    apply_rotation: False
    grid_size: 0.05
    mode: "last"

    train_pre_batch_collate_transform:
    - transform: ClampBatchSize
      params:
        num_points: 1000000

    train_transform:
    - transform: Random3AxisRotation
      params:
        apply_rotation: ${data.apply_rotation}
        rot_x: 8
        rot_y: 8
        rot_z: 180
    - transform: RandomSymmetry
      params:
        axis: [True, True, False]
    - transform: GridSampling3D
      params:
        size: ${data.grid_size}
        quantize_coords: True
        mode: ${data.mode}

    val_transform:
    - transform: GridSampling3D
      params:
        size: ${data.grid_size}
        quantize_coords: True
        mode: ${data.mode}
models:
    STRes16UNet14B:
        class: minkowski.Minkowski_Baseline_Model
        conv_type: "SPARSE"
        model_name: "STRes16UNet14B"
        D: 4
        backbone_out_dim: 8
        extra_options:
            conv1_kernel_size: 5
"""

from omegaconf import OmegaConf
params = OmegaConf.create(yaml_config)

from torch_points3d.datasets.grasp_classification import acronymvid
dataset = acronymvid.AcronymVidDataset(params.data)

model = Minkowski_Baseline_Model(option=params.models[params.model_name],
    model_type=None,
    dataset=dataset,
    modules=None)

dataset.create_dataloaders(model,
    batch_size=2,
    shuffle=False,
    num_workers=1,
    precompute_multi_scale=False
)

loader = dataset.test_dataloaders[0]

data = next(iter(loader))

import time
tic = time.time()

device = torch.device("cuda")
model = model.to(device)

model.set_input(data, device)
model.forward() # self.input.F.shape is [64315, 1]
model.backward()

print(f"done in time {time.time() - tic}")