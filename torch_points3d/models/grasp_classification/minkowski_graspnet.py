from functools import reduce
import logging
import torch.nn.functional as F
import torch
import numpy as np

from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_model import BaseModel


log = logging.getLogger(__name__)


class Minkowski_Baseline_Model(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model, self).__init__(option)
        self._weight_classes = dataset.weight_classes
        self.model = GraspNet(option, dataset)
        self.loss_names = ["loss_grasp", "add_s_loss", "classification_loss"]

    def set_input(self, data, device):

        self.data = data # store data as instance variable in RAM for visualization
        self.single_gripper_points = data[0].single_gripper_pts.to(device)

        # randomly downsample 4D points, such that each time step has num_points
        pts_per_frame = data.pos.shape[0] / len(data.batch.unique()) / len(data.time.unique()) # 90,000 pixels = 300 x 300
        assert self.opt.points_per_frame < pts_per_frame
        # always guarantee num_points per frame
        kept_idxs = []
        frame_num = 0
        for b in data.batch.unique():
            for t in data.time.unique():
                perm = torch.randperm(int(pts_per_frame), device=torch.device("cpu")) + frame_num*pts_per_frame
                kept_idxs.append(perm[:self.opt.points_per_frame])
                frame_num += 1

        self.idx, _ = torch.cat(kept_idxs).sort()
        self.idx = self.idx.long()
        # perm = torch.randperm(data.pos.shape[0])
        # self.idx = perm[:self.opt.num_points]
        # self.idx, _ = self.idx.sort()

        batch = data.batch[self.idx]
        time = data.time[self.idx]
        pos = data.pos[self.idx]
        x = data.x[self.idx]
        y = data.y[self.idx]

        # quantize position across a voxel grid, truncating decimals
        quantized_pos = (pos / self.opt.grid_size).int()

        coords = torch.cat([
            batch.reshape(-1, 1), 
            time.reshape(-1, 1), 
            quantized_pos,
            ], dim=1).int().to(device)

        features = x.to(device)

        self.input = ME.SparseTensor(
            features=features,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=device)

        self.labels = y.to(device)

        # Pack the ground truth control points into a dense tensor.
        # If the control point tensors are not the same shape, then duplicate entries from the smaller ones until they are the same shape. Because the ADD-S loss considers the minimum distance from a ground truth grasp, duplicating ground truth grasps will not affect the grasp.

        cp_shapes = [cp.shape for cp in data.pos_control_points]
        num_pos_grasps = [shape[1] for shape in cp_shapes]
        self.pos_control_points = torch.empty((
            len(batch.unique()),
            len(time.unique()),
            max(num_pos_grasps),
            5,
            3
        ), device=device)
        self.sym_pos_control_points = torch.empty((
            len(batch.unique()),
            len(time.unique()),
            max(num_pos_grasps),
            5,
            3
        ), device=device)

        # Pad control point tensors by repeating if different.
        if not reduce(np.array_equal, cp_shapes):
            for i in range(len(data.pos_control_points)):
                if num_pos_grasps[i] > 0:
                    idxs = list(range(num_pos_grasps[i]))
                    idxs = idxs + [0]*(max(num_pos_grasps) - num_pos_grasps[i])
                    self.pos_control_points[i] = torch.from_numpy(data.pos_control_points[i][:,idxs,:,:]).to(device)
                    self.sym_pos_control_points[i] = torch.from_numpy(data.sym_pos_control_points[i][:,idxs,:,:]).to(device)
        else:
            for i in range(len(data.pos_control_points)):
                if num_pos_grasps[i] > 0:
                    self.pos_control_points[i] = torch.from_numpy(data.pos_control_points[i]).to(device)
                    self.sym_pos_control_points[i] = torch.from_numpy(data.sym_pos_control_points[i]).to(device)

        # Store 3d positions corresponding to coordinates
        self.positions = torch.Tensor(pos).to(device)

    def forward(self, *args, **kwargs):
        self.class_logits, self.approach_dir, self.baseline_dir, self.grasp_width = self.model(self.input)

    def _compute_losses(self):

        self.add_s_loss = add_s_loss(
            approach_dir = self.approach_dir, 
            baseline_dir = self.baseline_dir, 
            coords = self.input.coordinates[self.input.inverse_mapping], 
            positions = self.positions,
            pos_control_points = self.pos_control_points,
            sym_pos_control_points = self.sym_pos_control_points,
            single_gripper_points = self.single_gripper_points,
            labels = self.labels,
            logits = self.class_logits,
            grasp_width = self.grasp_width,
            device = self.device)

        self.classification_loss = F.binary_cross_entropy_with_logits(
            self.class_logits,
            self.labels
        )

        self.loss_grasp = self.opt.bce_loss_coeff*self.classification_loss + self.opt.add_s_loss_coeff*self.add_s_loss 

    def backward(self):
        self._compute_losses()
        self.loss_grasp.backward()

def sequential_add_s_loss(approach_dir, baseline_dir, coords, positions, pos_control_points, sym_pos_control_points, single_gripper_points, labels, logits, grasp_width, device) -> torch.Tensor:
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

    logits = logits.reshape((n_batch, n_time, -1))
    labels = labels.reshape((n_batch, n_time, -1))
    loss = torch.zeros(1, device=device)
    for b in range(n_batch):
        for t in range(n_time):
            ## Find the minimum pairwise distance from each predicted grasp to the ground truth grasps.
            dists = approx_min_dists(
                pred_cp[b][t][None, None, :], 
                pos_control_points[b][t][None, None, :]
            )

            loss += torch.mean(
                torch.sigmoid(logits[b][t]) *   # weight by confidence
                labels[b][t] *                  # only backprop positives
                dists.view(-1)                  # weight by distance
            )
    return loss

def add_s_loss(approach_dir, baseline_dir, coords, positions, pos_control_points, sym_pos_control_points, single_gripper_points, labels, logits, grasp_width, device) -> torch.Tensor:
    
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
    dists = approx_min_dists(pred_cp, pos_control_points)

    loss = torch.mean(
        torch.sigmoid(logits) * # weight by confidence
        labels *                # only backprop positives
        dists.view(-1)           # weight by distance
    )
    return loss

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

def old_add_s_loss(approach_dir, baseline_dir, coords, positions, pos_control_points, sym_pos_control_points, single_gripper_points, labels, logits, grasp_width, device) -> torch.Tensor:
    """Compute add-s loss from Contact-GraspNet.

    The add-s loss is the"minimum average distance from a predicted grasp's
    control points to the control points of ground truth grasps.

    Args: approach_dir (torch.Tensor): (M, 3) tensor of "approach vectors"
        baseline_dir (torch.Tensor): (M, 3) tensor of "baseline vectors" coords
        (torch.Tensor): (M, 5) tensors of input coordinates for the M vectors
        pos_control_points (torch.Tensor): (batchsize, ntimes, n_gt_grasps, 5,
        3) tensor of control points

    Returns: torch.Tensor: (B,) tensor of scalar losses
    """
    # self.pos_control_points.shape is (2, 10, 1317, 5, 3)
    pred_grasp_tfs = build_6dof_grasps(positions, baseline_dir, approach_dir, grasp_width, device)
    
    # calculate the SE(3) transforms corresponding to each predicted approach/baseline pair.
    
    # determine the control points of the predicted grasps for each point
    single_gripper_points_homog = torch.cat([
        single_gripper_points, 
        torch.ones((len(single_gripper_points), 1), device=device)],
        dim=1
    )
    pred_control_pts = torch.matmul(single_gripper_points_homog, torch.transpose(pred_grasp_tfs, 2, 1))[:,:,:3]

    # There are potentially different numbers of 3D points at each time step.
    # For now, we'll commit the cardinal sin of parallel GPU programming: a for loop

    times = coords[:,1].unique()
    batches = coords[:,0].unique()

    # # Later, we can try to convert to operate on a single rectangular
    # # multi-dimensional tensor, padded by Nan.
    # times, counts = coords[:,1].unique(return_counts=True)
    # most_points = counts.max() # highest number of 3D points in a given time step
    # batches = coords[:,0].unique()
    # # form a regular tensor of dimension
    # # (batch, time, max_num_points, 5, 3)
    # # padded by NaNs where the num_points is less than max_num_points
    # pred_cp = float('nan') * torch.ones((len(batches), len(times), most_points, 5, 3))

    # generate mapping from coordinates to integers

    loss = torch.zeros(1, device=device)
    for i, b in enumerate(batches):
        for j, t in enumerate(times):
            
            idxs = torch.where(torch.logical_and(coords[:,1] == t, coords[:,0] == b))

            # get predicted control points for this frame: this timestep, at this batch
            pred_cp_frame = pred_control_pts[idxs] # (2973, 5, 3)

            gt_cp_frame = pos_control_points[i][j, :, :, :] # (1317, 5, 3)
            sym_gt_cp_frame = sym_pos_control_points[i][j, :, :, :] # (1317, 5, 3)

            if gt_cp_frame.shape[0] == 0:
                loss += 0.0
                break

            neg_squared_add_k = neg_squared_add_k_fn(pred_cp_frame, gt_cp_frame, sym_gt_cp_frame)

            labels_frame = labels[idxs]
            logits_frame = logits[idxs]

            loss_frame = torch.mean(
                torch.sigmoid(logits_frame.squeeze()) *     # weight by confidence
                labels_frame.squeeze() *                    # only backprop positives
                torch.sqrt(-neg_squared_add_k.squeeze())    # weight by distance
            )
                
            loss += loss_frame

    return loss / (len(batches) * len(times)) # take average

def orig_neg_squared_add_k_fn(pred_cp_frame, gt_cp_frame, sym_gt_cp_frame):
    squared_add = torch.sum((torch.unsqueeze(pred_cp_frame, 1) - torch.unsqueeze(gt_cp_frame, 0))**2, dim=(2, 3))
    sym_squared_add = torch.sum((torch.unsqueeze(pred_cp_frame, 1) - torch.unsqueeze(sym_gt_cp_frame, 0))**2, dim=(2, 3))
    ## (n_grasp, n_gt_grasps) 

    neg_squared_add = -torch.cat([squared_add, sym_squared_add], dim=1) # (n_grasp, 2*n_gt_grasp)

    neg_squared_add_k = torch.topk(neg_squared_add, k=1, sorted=False, dim=1)[0] # (n_grasp)

    return neg_squared_add_k

@torch.jit.script
def neg_squared_add_k_fn(pred_cp_frame, gt_cp_frame, sym_gt_cp_frame):
    gt_cp_means = gt_cp_frame.mean(dim=1)
    sym_gt_cp_means = sym_gt_cp_frame.mean(dim=1)
    pred_cp_means = pred_cp_frame.mean(dim=1)

    squared_add = torch.sum((torch.unsqueeze(pred_cp_means, 1) - torch.unsqueeze(gt_cp_means, 0))**2, dim=2)
    sym_squared_add = torch.sum((torch.unsqueeze(pred_cp_means, 1) - torch.unsqueeze(sym_gt_cp_means, 0))**2, dim=2)

    neg_squared_add = -torch.cat([squared_add, sym_squared_add], dim=1) 

    best_idx = torch.topk(neg_squared_add, k=1, sorted=False, dim=1)[1].squeeze()
    all_cp_frame = torch.cat([gt_cp_frame, sym_gt_cp_frame], dim=0)
    neg_squared_add_k =  -torch.sum((pred_cp_frame - all_cp_frame[best_idx])**2, dim=(1,2))
    return neg_squared_add_k

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