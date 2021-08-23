import os, sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

import torch
from torch_points3d.models.grasp_classification.minkowski_graspnet import Minkowski_Baseline_Model
# from torch_points3d.models.grasp_classification.minkowski import Minkowski_Baseline_Model

yaml_config = """
model_name: GraspMinkUNet14A
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
  GraspMinkUNet14A:
    class: minkowski_graspnet.Minkowski_Baseline_Model
    conv_type: "SPARSE"
    model_name: "MinkUNet14A"
    D: 4
    backbone_out_dim: 128
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
    num_workers=4,
    precompute_multi_scale=False
)

loader = dataset.test_dataloaders[0]

import time
tic = time.time()

device = torch.device("cuda")
# device = torch.device("cpu")
model = model.to(device)
t0a = time.time()
print(f"Time to push model to device: \t{t0a - tic}")

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.2)
t0b = time.time()
print(f"Time to initialize optimizer: \t{t0b - t0a}")

input_times = []
fwd_times = []
bwd_times = []
tot_times = []

for i in range(10):

  data = next(iter(loader))
  # optimizer.zero_grad()

  t0 = time.time()
  model.set_input(data, device)

  t1 = time.time()
  input_times.append(t1 - t0)

  model.forward() # self.input.F.shape is [64315, 1]
  
  t2 = time.time()
  fwd_times.append(t2 - t1)

  model.backward()
  optimizer.step()

  t3 = time.time()
  bwd_times.append(t3 - t2)
    
  tot_times.append(t3 - t0)


  print(model.add_s_loss.item())

import numpy as np
print(f"Input Time: \t{np.mean(input_times)} \t+/- {np.std(input_times)}")
print(f"fwd Time: \t{np.mean(fwd_times)} \t+/- {np.std(fwd_times)}")
print(f"bwd Time: \t{np.mean(bwd_times)} \t+/- {np.std(bwd_times)}")
print(f"tot Time: \t{np.mean(tot_times)} \t+/- {np.std(tot_times)}")
  

print(f"done in time {time.time() - tic}")