from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.metrics.base_tracker import BaseTracker

import h5py
import os
import numpy as np
import torch

from torch_geometric.data import Dataset, Data

class AcronymVidDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        process_workers: int = dataset_opt.process_workers if dataset_opt.process_workers else 0

        self.train_dataset = GraspDataset(self._data_path, split="train", process_workers=process_workers)

        self.test_dataset = GraspDataset(self._data_path, split="test", process_workers=process_workers)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return BaseTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

class GraspDataset(Dataset):
    """
    A torch.geometric.Dataset for loading from files.
    """

    AVAILABLE_SPLITS = ["train", "val", "test"]

    def __init__(self, root, split="train", transform=None, process_workers=1, pre_transform=None):

        self.use_multiprocessing = process_workers > 1
        self.process_workers = process_workers

        # Find the raw filepaths. For now, we're doing no file-based preprocessing.
        if split in ["train", "val", "test"]:
            folder = os.path.join(root, split)
            self._paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')]
        else:
            raise ValueError("Split %s not recognised" % split)

        # Make a list of tuples (name, path) for each trajectory.
        self._trajectories = []
        for path in self._paths:
            with h5py.File(path) as ds:
                keys = {k for k in ds.keys() if k.startswith('pitch')} # a Set
            self._trajectories += [(k, path) for k in keys]

        super().__init__(root, transform=transform, pre_transform=pre_transform)

    def download(self):
        if len(os.listdir(self.raw_dir)) == 0:
            print(f"No files found in {self.raw_dir}. Please create the dataset using the scripts in GraspRefinement and ACRONYM.")

    def len(self):
        return len(self._trajectories)

    def get(self, idx):
        traj_name, path = self._trajectories[idx]

        with h5py.File(path) as ds:
            # `depth` and `labels` are (B, 300,300) arrays.
            # `depth` contains the depth video values, and `labels` is a binary mask indicating
            # whether a given pixel's 3D point is within data_generation.params.EPSILON of a positive grasp contact.
            depth = np.asarray(ds[traj_name]["depth"])
            labels = np.asarray(ds[traj_name]["grasp_labels"])

        # make data shorter
        depth = depth[::3, :, :]
        labels = labels[::3, :, :]

        pcs = [depth_to_pointcloud(d) for d in depth]
        pcs = multi_pointcloud_to_4d_coords(pcs)

        data = Data(
            coords=torch.Tensor(pcs), 
            x=torch.ones((len(pcs), 1)), 
            y=torch.Tensor(labels).view(-1 ,1)
        )

        return data

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        # `self._trajectories` is a list of tuples (traj_name, path)
        paths = [os.path.basename(t[1]) for t in self._trajectories]
        return list(set(paths)) # remove duplicates

    @property
    def num_features(self):
        return 1

    @property
    def num_classes(self):
        return 2

def multi_pointcloud_to_4d_coords(pcs):
    """Convert a list of N (L, 3) ndarrays into a single (N*L, 4) ndarray, where the first dimension becomes the time dimension."""
    num_pcs = len(pcs)
    time_coords = np.repeat(np.arange(num_pcs), len(pcs[0]))
    
    pcs = np.concatenate(pcs)
    pcs = np.column_stack([time_coords, pcs])

    return pcs

def depth_to_pointcloud(depth, fov=np.pi/6):
    """Convert depth image to pointcloud given camera intrinsics, from acronym.scripts.acronym_render_observations

    Args:
        depth (np.ndarray): Depth image.

    Returns:
        np.ndarray: Point cloud.
    """
    fy = fx = 0.5 / np.tan(fov * 0.5)  # aspectRatio is one.

    height = depth.shape[0]
    width = depth.shape[1]

    mask = np.where(depth > 0)

    x = mask[1]
    y = mask[0]

    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = depth[y, x]

    return np.vstack((world_x, world_y, world_z)).T

if __name__ == "__main__":
    # Test loading dataset.
    gds = GraspDataset(root="/home/tim/Research/GraspRefinement/data/acronymvid", split="test")

    print(gds)
    print(gds[0])

    breakpoint()

    print("done")