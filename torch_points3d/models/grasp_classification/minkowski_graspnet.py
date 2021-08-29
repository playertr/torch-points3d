from functools import reduce
import logging
import torch.nn.functional as F
import torch
import numpy as np
import random

from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_model import BaseModel
import torch_geometric

log = logging.getLogger(__name__)

class Minkowski_Baseline_Model(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model, self).__init__(option)
        self._weight_classes = dataset.weight_classes
        self.model = GraspNet(option, dataset)
        self.loss_names = ["loss_grasp", "add_s_loss", "classification_loss"]

    def set_input(self, data, device):

        self.data = data # store data as instance variable in RAM for debug

        ## Randomly downsample 4D points, such that each time step has num_points

        batch, time, pos, x, y, self.idx = downsample(data, self.opt.points_per_frame, device)
            
        ## Quantize position across a voxel grid, truncating decimals
        quantized_pos = (pos / self.opt.grid_size).int()

        ## Convert 4D coordinates and features to SparseTensor
        coords = torch.cat([
            batch.view(-1, 1), 
            time.view(-1, 1), 
            quantized_pos,
            ], dim=1).int()
        features = x
        self.input = ME.SparseTensor(
            features=features,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=device)

        self.pos_control_points, self.sym_pos_control_points, self.gt_grasps_per_batch = collate_control_points(data, device)

        self.single_gripper_points = data[0].single_gripper_pts.to(device)
        
        self.positions = pos # Store 3d positions corresponding to coordinates

        self.labels = y

    def forward(self, *args, **kwargs):
        self.class_logits, self.approach_dir, self.baseline_dir, self.grasp_width = self.model(self.input)

    def _compute_losses(self):

        add_s = parallel_add_s_loss if self.opt.parallel_add_s else sequential_add_s_loss
        self.add_s_loss = add_s(
            approach_dir = self.approach_dir, 
            baseline_dir = self.baseline_dir, 
            coords = self.input.coordinates[self.input.inverse_mapping], 
            positions = self.positions,
            pos_control_points = self.pos_control_points,
            sym_pos_control_points = self.sym_pos_control_points,
            gt_grasps_per_batch = self.gt_grasps_per_batch,
            single_gripper_points = self.single_gripper_points,
            labels = self.labels,
            logits = self.class_logits,
            grasp_width = self.grasp_width,
            device = self.device
        )

        self.classification_loss = F.binary_cross_entropy_with_logits(
            self.class_logits,
            self.labels
        )

        self.loss_grasp = self.opt.bce_loss_coeff*self.classification_loss + self.opt.add_s_loss_coeff*self.add_s_loss 

    def backward(self):
        self._compute_losses()
        self.loss_grasp.backward()

def downsample(data: torch_geometric.data.Batch, des_pts_per_frame, device):
    """Randomly select a subset of pixels to pass to the network, such that each individual image has the same number of pixels: `des_pts_per_frame`.

    The data is given as (n_batch*n_time*300*300, D) tensors, where D is the trailing dimension of each item of interest in `data`. To generate indices for fast random selection on the GPU, we generate a 3D random tensor and argsort it along a single dimension (fast on GPU). This lets us quickly create different random indices simultaneously for all images from different batches and times.
     """

    n_batch = len(data.batch.unique())
    n_time = len(data.time.unique())

    # 90,000 pixels = 300 x 300
    pts_per_frame = int(data.pos.shape[0] / n_batch / n_time) 
    assert des_pts_per_frame < pts_per_frame

    inds = torch.argsort(
        torch.rand((n_batch, n_time, pts_per_frame), device=device), 
        dim=2)[:,:,:des_pts_per_frame]

    def index_1d_tensor(t, inds):
        """Turn a 1D tensor into a (n_batch, n_time, pts_per_frame) 3D tensor, then apply the 3D indexing. Return flattened output."""
        t = t.view(n_batch, n_time, -1)
        t = t[
            torch.arange(n_batch).view(-1, 1, 1),
            torch.arange(n_time).view(1, -1, 1),
            inds 
        ]
        return t.view(-1, 1)
    
    def index_3d_tensor(t, inds):
        """Turn a (n_batch*n_time*pts_per_frame, 3) tensor into a (n_batch, n_time, pts_per_frame, 3) 4D tensor, then apply the 3D indexing. Return flattened output."""
        t = t.view(n_batch, n_time, -1, 3)
        t = t[
            torch.arange(n_batch).view(-1, 1, 1, 1),
            torch.arange(n_time).view(1, -1, 1, 1),
            inds.unsqueeze(-1),
            torch.arange(3).view(1, 1, 1, -1) 
        ]
        return t.view(-1, 3)
    
    batch = index_1d_tensor(data.batch.to(device), inds)
    time = index_1d_tensor(data.time.to(device), inds)
    pos = index_3d_tensor(data.pos.to(device), inds)
    x = index_1d_tensor(data.x.to(device), inds).view(-1, 1)
    y = index_1d_tensor(data.y.to(device), inds).view(-1, 1)

    return batch, time, pos, x, y, inds

def collate_control_points(data, device):
    """Pack the ground truth control points into a dense tensor.

    If the control point tensors are not the same shape, then duplicate entries from the smaller ones until they are the same shape. Because the ADD-S loss considers the minimum distance from a ground truth grasp, duplicating ground truth grasps will not affect the grasp.
    """

    n_batch = len(data.batch.unique())
    n_time = len(data.time.unique())

    cp_shapes = [cp.shape for cp in data.pos_control_points]
    gt_grasps_per_batch = [shape[1] for shape in cp_shapes]
    pos_control_points = torch.empty((
        n_batch,
        n_time,
        max(gt_grasps_per_batch),
        5,
        3
    ), device=device)
    sym_pos_control_points = torch.empty((
        n_batch,
        n_time,
        max(gt_grasps_per_batch),
        5,
        3
    ), device=device)

    # Pad control point tensors by repeating if different.
    if not reduce(np.array_equal, cp_shapes):
        for i in range(len(data.pos_control_points)):
            if gt_grasps_per_batch[i] > 0:
                idxs = list(range(gt_grasps_per_batch[i]))
                idxs = idxs + [0]*(max(gt_grasps_per_batch) - gt_grasps_per_batch[i])
                pos_control_points[i] = torch.from_numpy(data.pos_control_points[i][:,idxs,:,:]).to(device)
                sym_pos_control_points[i] = torch.from_numpy(data.sym_pos_control_points[i][:,idxs,:,:]).to(device)
            # else: leave tensor uninitialized; do not use in ADD-S loss.
    else:
        for i in range(len(data.pos_control_points)):
            if gt_grasps_per_batch[i] > 0:
                pos_control_points[i] = torch.from_numpy(data.pos_control_points[i]).to(device)
                sym_pos_control_points[i] = torch.from_numpy(data.sym_pos_control_points[i]).to(device)
            # else: leave tensor uninitialized; do not use in ADD-S loss.
            
    return pos_control_points, sym_pos_control_points, gt_grasps_per_batch

def parallel_add_s_loss(approach_dir, baseline_dir, coords, positions, pos_control_points, sym_pos_control_points, gt_grasps_per_batch, single_gripper_points, labels, logits, grasp_width, device) -> torch.Tensor:
    """Compute symmetric ADD-S loss from Contact-GraspNet, finding minimum control-point distances for all time and batch values in parallel."""
    
    if min(gt_grasps_per_batch) == 0:
        # one of the batches has no ground truth grasps; fall back to non-parallelized ADD-S computation so that the other batches can still contribute to the loss.
        return sequential_add_s_loss(approach_dir, baseline_dir, coords, positions, pos_control_points, sym_pos_control_points, gt_grasps_per_batch, single_gripper_points, labels, logits, grasp_width, device)

    ## Package each grasp parameter P into a regular, dense Tensor of shape
    # (BATCH, TIME, N_PRED_GRASP, *P.shape)
    n_batch = len(coords[:,0].unique())
    n_time = len(coords[:,1].unique())
    approach_dir = approach_dir.view(n_batch, n_time, -1, 3)
    baseline_dir = baseline_dir.view(n_batch, n_time, -1, 3)
    positions = positions.view(n_batch, n_time, -1, 3)
    grasp_width = grasp_width.view(n_batch, n_time, -1, 1)

    ## Construct control points for the predicted grasps, where label is True.
    pred_cp = control_point_tensor(
        approach_dir,
        baseline_dir,
        positions,
        grasp_width,
        single_gripper_points,
        device
    )

    ## Find the minimum pairwise distance from each predicted grasp to the ground truth grasps.
    dists = torch.minimum(
        approx_min_dists(pred_cp, pos_control_points),
        approx_min_dists(pred_cp, sym_pos_control_points)
    )

    loss = torch.mean(
        torch.sigmoid(logits) * # weight by confidence
        labels *                # only backprop positives
        dists.view(-1)          # weight by distance
    )
    return loss

def sequential_add_s_loss(approach_dir, baseline_dir, coords, positions, pos_control_points, sym_pos_control_points, gt_grasps_per_batch, single_gripper_points, labels, logits, grasp_width, device) -> torch.Tensor:
    """Un-parallelized implementation of add_s_loss from below. Uses a loop instead of batch/time parallelization to reduce memory requirements. """

    ## Package each grasp parameter P into a regular, dense Tensor of shape
    # (BATCH, TIME, N_PRED_GRASP, *P.shape)
    n_batch = len(coords[:,0].unique())
    n_time = len(coords[:,1].unique())
    approach_dir = approach_dir.view(n_batch, n_time, -1, 3)
    baseline_dir = baseline_dir.view(n_batch, n_time, -1, 3)
    positions = positions.view(n_batch, n_time, -1, 3)
    grasp_width = grasp_width.view(n_batch, n_time, -1, 1)

    ## Construct control points for the predicted grasps, where label is True.
    pred_cp = control_point_tensor(
        approach_dir,
        baseline_dir,
        positions,
        grasp_width,
        single_gripper_points,
        device
    )

    logits = logits.view((n_batch, n_time, -1))
    labels = labels.view((n_batch, n_time, -1))
    loss = torch.zeros(1, device=device)
    for b in range(n_batch):
        
        if gt_grasps_per_batch[b] == 0:
            # This has no ground truth grasps; do not backprop ADD-S loss for this batch.
            loss += 0.0
            break

        for t in range(n_time):
            ## Find the minimum pairwise distance from each predicted grasp to the ground truth grasps.

            dists = torch.minimum(
                approx_min_dists(
                    pred_cp[b][t][None, None, :], 
                    pos_control_points[b][t][None, None, :]
                ),
                approx_min_dists(
                pred_cp[b][t][None, None, :], 
                sym_pos_control_points[b][t][None, None, :]
                )
            )

            loss += torch.mean(
                torch.sigmoid(logits[b][t]) *   # weight by confidence
                labels[b][t] *                  # only backprop positives
                dists.view(-1)                  # weight by distance
            )
    return loss / (n_batch * n_time)

def approx_min_dists(pred_cp, gt_cp):
    """Find the approximate minimum average distance of each control-point-tensor in `pred_cp` from any of the control-point-tensors in `gt_cp`.

    Approximates distance by finding the mean three-dimensional coordinate of each tensor, so that the M x N pairwise lookup takes up one-fifth of the memory as comparing all control-point-tensors in full, avoiding the creation of a (B, T, N, M, 5, 3) matrix.
    
    Once the mean-closest tensor is found for each point, the full matrix L2 distance is returned.

    Args:
        pred_cp (torch.Tensor): (B, T, N, 5, 3) Tensor of control points
        gt_cp (torch.Tensor): (B, T, M, 5, 3) Tensor of ground truth control points
    """
    # Take the mean 3-vector from each 5x3 control point Tensor
    m_pred_cp = pred_cp.mean(dim=3) # (B, T, N, 3)
    m_gt_cp = gt_cp.mean(dim=3)     # (B, T, M, 3)

    # Find the squared L2 distance between all N pred and M gt means.
    # Find the index of the ground truth grasp minimizing the L2 distance.
    approx_sq_dists = torch.sum((m_pred_cp.unsqueeze(3) - m_gt_cp.unsqueeze(2))**2, dim=4)  # (B, T, N, M)
    best_idxs = torch.topk(-approx_sq_dists, k=1, sorted=False, dim=3)[1]   # (B, T, N, 1)

    # Select the full 5x3 matrix corresponding to each minimum-distance grasp.
    # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    closest_gt_cps = torch.gather(gt_cp, dim=2, index=best_idxs.unsqueeze(4).repeat(1, 1, 1, 5, 3)) # (B, T, N, 5, 3)

    # Find the matrix L2 distances
    best_l2_dists = torch.sqrt(torch.sum((pred_cp - closest_gt_cps)**2, dim=(3,4))) # (B, T, N)

    return best_l2_dists

def control_point_tensor(approach_dirs, baseline_dirs, positions, grasp_widths, gripper_pts, device):
    """Construct an (N, 5, 3) Tensor of "gripper control points". From Contact-GraspNet.

    Each of the N grasps is represented by five 3-D control points.

    Args:
        approach_dirs (torch.Tensor): (B, T, N, 3) set of unit approach directions.
        baseline_dirs (torch.Tensor): (B, T, N, 3) ser of unit gripper axis directions.
        positions (torch.Tensor): (B, T, N, 3) set of gripper contact points.
        grasp_widths (torch.Tensor): (B, T, N) set of grasp widths
        gripper_pts (torch.Tensor): (5,3) Tensor of gripper-frame points
    """
    # Retrieve 6-DOF transformations
    grasp_tfs = build_6dof_grasps(
        contact_pts=positions,
        baseline_dir=baseline_dirs,
        approach_dir=approach_dirs,
        grasp_width=grasp_widths,
        device=device
    )
    # Transform the gripper-frame points into camera frame
    gripper_pts = torch.cat([
        gripper_pts, 
        torch.ones((len(gripper_pts), 1), device=device)],
        dim=1
    ) # make (5, 4) stack of homogeneous vectors

    pred_cp = torch.matmul(
        gripper_pts, 
        torch.transpose(grasp_tfs, 4, 3)
    )[...,:3]

    return pred_cp

def build_6dof_grasps(contact_pts, baseline_dir, approach_dir, grasp_width, device, gripper_depth=0.1034):
    """Calculate the SE(3) transforms corresponding to each predicted coord/approach/baseline/grasp_width grasp.
    """
    grasps_R = torch.stack([baseline_dir, torch.cross(approach_dir, baseline_dir), approach_dir], axis=4)
    grasps_t = contact_pts + grasp_width/2 * baseline_dir - gripper_depth * approach_dir
    ones = torch.ones((*contact_pts.shape[:3], 1, 1), device=device)
    zeros = torch.zeros((*contact_pts.shape[:3], 1, 3), device=device)
    homog_vec = torch.cat([zeros, ones], axis=4)

    pred_grasp_tfs = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, 4)], dim=4), homog_vec], dim=3)
    return pred_grasp_tfs

class GraspNet(torch.nn.Module):
    def __init__(self, option, dataset):
        super().__init__()
        self.backbone = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, option.backbone_out_dim, D=option.D, **option.get("extra_options", {})
        )
        self.classification_head = nn.Sequential(
            nn.Conv1d(in_channels=option.backbone_out_dim, out_channels=128, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1),
        )
        self.baseline_dir_head = nn.Sequential(
            nn.Conv1d(in_channels=option.backbone_out_dim, out_channels=128, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1),
        )
        self.approach_dir_head = nn.Sequential(
            nn.Conv1d(in_channels=option.backbone_out_dim, out_channels=128, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1),
        )

        self.grasp_offset_head = nn.Sequential(
            nn.Conv1d(in_channels=option.backbone_out_dim, out_channels=128, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1),
        )
    
    def forward(self, sparse_x):
        """Accepts a Minkowski sparse tensor."""
        # B x 10 x ~300 x ~300
        torch.cuda.empty_cache()
        x = self.backbone(sparse_x)
        torch.cuda.empty_cache()

        x = x.slice(sparse_x).F
        torch.cuda.empty_cache()

        class_logits = self.classification_head(x.unsqueeze(-1)).squeeze(dim=-1)

        # Gram-Schmidt normalization
        baseline_dir = self.baseline_dir_head(x.unsqueeze(-1)).squeeze()
        baseline_dir = F.normalize(baseline_dir)

        approach_dir = self.approach_dir_head(x.unsqueeze(-1)).squeeze()
        dot_product =  torch.sum(baseline_dir*approach_dir, dim=-1, keepdim=True)
        approach_dir = F.normalize(approach_dir - dot_product*baseline_dir)

        grasp_offset = F.relu(self.grasp_offset_head(x.unsqueeze(-1)).squeeze(dim=-1))

        return class_logits, approach_dir, baseline_dir, grasp_offset
