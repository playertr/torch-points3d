import sys
DIR = "/home/tim/Research/torch-points3d/"
sys.path.insert(0, DIR)

from torch_points3d.datasets.grasp_classification import acronymvid
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

import torch
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

def success_from_labels(pred_classes, actual_classes):
    """Return success, as proportion, from ground-truth point labels.
    
    (correct positive predictions) / (total positive predictions)"""
    all_positives = pred_classes == 1
    true_positives = np.logical_and(all_positives, (pred_classes == actual_classes))
    return true_positives.sum() / all_positives.sum()

def coverage(gt_classes, pred_grasps, gt_grasps, radius=0.005):
    """Get the proportion of ground truth grasps within epsilon of predicted.
    
    (gt close to predicted) / (total gt)
    """
    if len(pred_grasps) == 0: return np.nan

    # for each grasp coordinate in gt_grasps, find the distance to every grasp grasp_coordinate in pred_grasps
    pos_gt_grasps = gt_grasps[np.where(gt_classes.squeeze() == 1)]
    dists = distances(pos_gt_grasps, pred_grasps) # (n_gt, n_pred)
    closest_dists = dists.min(axis=1)
    return np.mean(closest_dists < radius)

def distances(grasps1, grasps2):
    """Finds the L2 distances from points in grasps1 to points in grasps2.

    Returns a (grasp1.shape[0], grasps3.shape[0]) ndarray."""

    diffs = np.expand_dims(grasps1, 1) - np.expand_dims(grasps2, 0)
    return np.linalg.norm(diffs, axis=-1)

def success_coverage_curve(confs, pred_grasps, gt_grasps, gt_labels):
    """Determine the success and coverage at various threshold confidence values."""

    res = []
    thresholds = np.linspace(0, 1, 100)
    for t in thresholds:
        pred_classes = confs > t
        pred_pos_grasps = pred_grasps[np.where(pred_classes.squeeze() == 1)]

        res.append({
            "confidence": t,
            "success": success_from_labels(pred_classes, gt_labels),
            "coverage": coverage(gt_labels, pred_pos_grasps, gt_grasps)
        })
    
    return pd.DataFrame(res)

def create_dataset(yaml_config=None):
    if yaml_config is None:

        yaml_config = """
        data:
            task: grasp_classification
            class: acronymvid.AcronymVidDataset
            name: acronymvid
            dataroot: /home/tim/Research/GraspRefinement/data
            process_workers: 8
            apply_rotation: False
            grid_size: 0.03
            mode: "last"

            train_pre_batch_collate_transform:
            -   transform: ClampBatchSize
                params:
                    num_points: 1000000

            train_transform:
            -   transform: Random3AxisRotation
                params:
                    apply_rotation: ${data.apply_rotation}
                    rot_x: 8
                    rot_y: 8
                    rot_z: 180
            -   transform: RandomSymmetry
                params:
                    axis: [True, True, False]
            -   transform: GridSampling3D
                params:
                    size: ${data.grid_size}
                    quantize_coords: True
                    mode: ${data.mode}

            val_transform:
            -   transform: GridSampling3D
                params:
                    size: ${data.grid_size}
                    quantize_coords: True
                    mode: ${data.mode}
        """
    params = OmegaConf.create(yaml_config)
    dataset = acronymvid.AcronymVidDataset(params.data)
    return dataset

def model_and_loader(ckpt_dir, check_name, device, yaml_config=None):
    """Create a pretrained model and Dataloader."""

    dataset = create_dataset(yaml_config)
    model_ckpt = ModelCheckpoint(ckpt_dir, check_name, "test", resume=False)
    model = model_ckpt.create_model(dataset, weight_name="best").to(device)
    model.eval()
    dataset.create_dataloaders(model,
        batch_size=2,
        shuffle=False,
        num_workers=6,
        precompute_multi_scale=False
    )
    loader = dataset.test_dataloaders[0]
    return model, loader

def plot_s_c_curve(df, ax=None, title="Coverage vs. Success"):
    if ax is None:
        fig, ax = plt.subplots()
    
    plot = ax.plot(df['coverage'], df['success'])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Success")
    ax.set_title(title)

    return plot

def plot_success_coverage_curve_on_testset():
    """ Plots success/coverage curve."""

    # Define terms the user might want to change
    device = torch.device("cuda")
    # device = torch.device("cpu")
    ckpt_dir = "/home/tim/Research/torch-points3d/outputs/2021-08-21/14-39-53"
    check_name = "GraspMinkUNet14A"

    # Create dataloader and model
    model, loader = model_and_loader(ckpt_dir, check_name, device)

    # Generate success/coverage curves for each batch in the test dataset
    s_c_curves = []
    for i, data in enumerate(tqdm(loader)):
        # Process a single batch
        with torch.no_grad():
            model.set_input(data, device)
            model.forward()

        # At each of the (batch, time, x, y, z) coordinates, get the output
        # confidence and ground truth label
        # TODO: change from "coords" (integer) to "pos" (float)
        out_coords = model.input.coordinates.detach().cpu().numpy()
        out_confs = torch.sigmoid(model.class_logits.detach()).cpu().numpy()
        grasp_labels = model.labels.detach().cpu().numpy()

        batches = np.unique(out_coords[:,0]) # batch dim is first column
        times = np.unique(out_coords[:,1]) # time is second column
        for b in batches:
            for t in times:
                idxs = np.logical_and(out_coords[:,0] == b, out_coords[:,1] ==t)
                pos = out_coords[idxs][:,2:] # position is last three columns
                confs = out_confs[idxs]
                labels = grasp_labels[idxs]

                s_c_curves.append(success_coverage_curve(confs, pos, pos, labels))

    print(s_c_curves[-1])
    plot_s_c_curve(s_c_curves[-1], title="Last Example")

    super_df = pd.concat(s_c_curves)
    super_df = super_df.groupby('confidence').mean()
    plot_s_c_curve(super_df, title="Entire Test Set")
    plt.show()
    breakpoint()
    print("done")

if __name__ == "__main__":
    plot_success_coverage_curve_on_testset()
