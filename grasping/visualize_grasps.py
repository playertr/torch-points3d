import os, sys
ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points3d.models.grasp_classification.minkowski_graspnet import build_6dof_grasps

from grasping.metric_utils import model_and_loader

import sys
CGN_DIR = "/home/tim/Research/contact_graspnet/"
sys.path.insert(0, CGN_DIR)
from contact_graspnet.visualization_utils import visualize_grasps

import torch
import numpy as np

def main():
    # Define terms the user might want to change
    device = torch.device("cuda")
    # device = torch.device("cpu")
    ckpt_dir = "/home/tim/Research/torch-points3d/outputs/2021-08-21/14-39-53"
    check_name = "GraspMinkUNet14A"

    # Create dataloader and model
    model, loader = model_and_loader(ckpt_dir, check_name, device)

    for i, data in enumerate(loader):
        with torch.no_grad():
            model.set_input(data, device)
            model.forward()

        # At each of the (batch, time, x, y, z) coordinates, get the output
        # confidence and ground truth label
        # TODO: change from "coords" (integer) to "pos" (float)
        out_coords = model.input.coordinates.detach().cpu()
        out_confs = torch.sigmoid(model.class_logits.detach()).cpu()
        out_baselines = model.baseline_dir.detach().cpu()
        out_approaches = model.approach_dir.detach().cpu()
        out_widths = model.grasp_width.detach().cpu()
        grasp_labels = model.labels.detach().cpu()

        pred_grasp_tfs = build_6dof_grasps(out_coords, out_baselines, out_approaches, out_widths, torch.device('cpu'))

        batches = np.unique(out_coords[:,0]) # batch dim is first column
        times = np.unique(out_coords[:,1]) # time is second column
        for b in batches:
            for t in times:
                idxs = np.where(
                    np.logical_and(out_coords[:,0] == b, out_coords[:,1] ==t)
                )
                pos = model.data.pos # position is last three columns
                confs = out_confs[idxs]
                labels = grasp_labels[idxs]
                tfs = pred_grasp_tfs[idxs]

                pred_pos_idxs = confs.squeeze() > 0.05
                pred_pos_tfs = tfs[pred_pos_idxs]
                pred_pos_confs = confs[pred_pos_idxs]

                breakpoint()

                visualize_grasps(
                    full_pc = pos.numpy(),
                    pred_grasps_cam = {0: pred_pos_tfs.numpy()},
                    scores = {0:pred_pos_confs.squeeze().numpy()},
                    pc_colors = np.ones(pos.shape)
                )
                
                break

if __name__ == "__main__":
    main()