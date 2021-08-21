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
        self.loss_names = ["loss_grasp"]

    def set_input(self, data, device):
        
        self.data = data # store data as instance variable in RAM for visualization
        self.single_gripper_points = torch.Tensor(data.single_gripper_pts[0]).to(self.device)

        torch.cuda.empty_cache()

        coords = torch.cat([data.batch.reshape(-1, 1), data.time.reshape(-1, 1), data.coords], dim=1).int()

        features = data.x.to(device=self.device)
        torch.cuda.empty_cache()

        self.input = ME.SparseTensor(
            features=features,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
            device=self.device)
        self.labels = data.y.to(self.device)

        # Identify ground truth grasps
        self.pos_control_points = data.pos_control_points # a list
        self.sym_pos_control_points = data.sym_pos_control_points

    def forward(self, *args, **kwargs):

        self.class_logits, self.approach_dir, self.baseline_dir = self.model(self.input)

        self.bce_loss = F.binary_cross_entropy_with_logits(self.class_logits,
            self.labels
        )

        self.add_s_loss = add_s_loss(
            self.approach_dir, 
            self.baseline_dir, 
            self.input.coordinates, 
            self.pos_control_points,
            self.sym_pos_control_points,
            self.single_gripper_points,
            self.labels,
            self.class_logits,
            self.device)

        self.classification_loss = F.binary_cross_entropy_with_logits(
            self.class_logits,
            self.labels
        )

        self.loss_grasp = self.classification_loss + 10*self.add_s_loss
        
    def backward(self):
        self.loss_grasp.backward()


def add_s_loss(approach_dir, baseline_dir, coords, pos_control_points, sym_pos_control_points, single_gripper_points, labels, logits, device) -> torch.Tensor:
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
    
    # calculate the SE(3) transforms corresponding to each predicted approach/baseline pair.
    gripper_width = 0.08 # from contact_graspnet/config.yaml
    gripper_depth = 0.1034

    # the last three columns of coords are position (NB! unstable Minkowski API)
    contact_pts = coords[:,2:]
    grasps_R = torch.stack([baseline_dir, torch.cross(approach_dir, baseline_dir), approach_dir], axis=2)
    grasps_t = contact_pts + gripper_width/2 * baseline_dir - gripper_depth * approach_dir
    ones = torch.ones((contact_pts.shape[0], 1, 1))
    zeros = torch.zeros((contact_pts.shape[0], 1, 3))
    homog_vec = torch.cat([zeros, ones], axis=2).to(device)

    pred_grasp_tfs = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, 2)], dim=2), homog_vec], dim=1)

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
            gt_cp_frame = torch.Tensor(pos_control_points[i][j, :, :, :]).to(device) # (1317, 5, 3)
            sym_gt_cp_frame = torch.Tensor(sym_pos_control_points[i][j, :, :, :]).to(device) # (1317, 5, 3)

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
            nn.Linear(option.backbone_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.approach_dir_head = nn.Sequential(
            nn.Linear(option.backbone_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        self.baseline_dir_head = nn.Sequential(
            nn.Linear(option.backbone_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
    
    def forward(self, sparse_x):
        """Accepts a Minkowski sparse tensor."""

        # B 3 10 x ~300 x ~300
        torch.cuda.empty_cache()
        x = self.backbone(sparse_x)
        torch.cuda.empty_cache()

        x = x.slice(sparse_x).F
        torch.cuda.empty_cache()

        class_logits = self.classification_head(x)

        approach_dir = self.approach_dir_head(x)

        baseline_dir = self.baseline_dir_head(x)

        return class_logits, approach_dir, baseline_dir