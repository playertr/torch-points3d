import logging
import torch.nn.functional as F
import torch

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

        # randomly downsample 4D points
        perm = torch.randperm(data.pos.shape[0])
        self.idx = perm[:self.opt.num_points]
        self.idx, _ = self.idx.sort()

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
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
            device=device)

        self.labels =y.to(device)

        # Identify ground truth grasps
        self.pos_control_points = [torch.Tensor(d).to(device) for d in data.pos_control_points]# a list
        self.sym_pos_control_points = [torch.Tensor(d).to(device) for d in data.sym_pos_control_points]

        # Store 3d positions corresponding to coordinates
        self.positions = torch.Tensor(pos).to(device)
        

    def forward(self, *args, **kwargs):

        self.class_logits, self.approach_dir, self.baseline_dir, self.grasp_width = self.model(self.input)

    def _compute_losses(self):

        self.add_s_loss = add_s_loss(
            approach_dir = self.approach_dir, 
            baseline_dir = self.baseline_dir, 
            coords = self.input.coordinates, 
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

def build_6dof_grasps(contact_pts, baseline_dir, approach_dir, grasp_width, device, gripper_depth=0.1034):
    """calculate the SE(3) transforms corresponding to each predicted coord/approach/baseline/grasp_width grasp.
    """
    grasps_R = torch.stack([baseline_dir, torch.cross(approach_dir, baseline_dir), approach_dir], axis=2)
    grasps_t = contact_pts + grasp_width/2 * baseline_dir - gripper_depth * approach_dir
    ones = torch.ones((contact_pts.shape[0], 1, 1), device=device)
    zeros = torch.zeros((contact_pts.shape[0], 1, 3), device=device)
    homog_vec = torch.cat([zeros, ones], axis=2)

    pred_grasp_tfs = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, 2)], dim=2), homog_vec], dim=1)
    return pred_grasp_tfs

def add_s_loss(approach_dir, baseline_dir, coords, positions, pos_control_points, sym_pos_control_points, single_gripper_points, labels, logits, grasp_width, device) -> torch.Tensor:
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

    loss = torch.Tensor([0.0]).to(device)
    for i, b in enumerate(batches):
        for j, t in enumerate(times):
            
            idxs = torch.where(torch.logical_and(coords[:,1] == t, coords[:,0] == b))

            # get predicted control points for this frame: this timestep, at this batch
            pred_cp_frame = pred_control_pts[idxs] # (2973, 5, 3)
            
            gt_cp_frame = pos_control_points[i][j, :, :, :] # (1317, 5, 3)
            sym_gt_cp_frame = sym_pos_control_points[i][j, :, :, :] # (1317, 5, 3)

            squared_add = torch.sum((torch.unsqueeze(pred_cp_frame, 1) - torch.unsqueeze(gt_cp_frame, 0))**2, dim=(2, 3))
            sym_squared_add = torch.sum((torch.unsqueeze(pred_cp_frame, 1) - torch.unsqueeze(sym_gt_cp_frame, 0))**2, dim=(2, 3))
            # (n_grasp, n_gt_grasps) 
            
            neg_squared_add = -torch.cat([squared_add, sym_squared_add], dim=1) # (n_grasp, 2*n_gt_grasp)

            try:
                neg_squared_add_k = torch.topk(neg_squared_add, k=1, sorted=False, dim=1)[0] # (n_grasp)
            except RuntimeError:
                # if an image has no labeled grasps, this triggers.
                loss += 0.0
                break

            labels_frame = labels[idxs]
            logits_frame = logits[idxs]

            loss_frame = torch.mean(
                torch.sigmoid(logits_frame.squeeze()) *     # weight by confidence
                labels_frame.squeeze() *                    # only backprop positives
                torch.sqrt(-neg_squared_add_k.squeeze())    # weight by distance
            )
                
            loss += loss_frame

    return loss / (len(batches) * len(times)) # take average

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

        # B 3 10 x ~300 x ~300
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

        grasp_offset = self.grasp_offset_head(x.unsqueeze(-1)).squeeze(dim=-1)

        return class_logits, approach_dir, baseline_dir, grasp_offset