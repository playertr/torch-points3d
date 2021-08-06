from enum import unique
import logging
import torch.nn.functional as F
import torch

from torch_points3d.modules.MinkowskiEngine import *
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.minkowski import Minkowski


log = logging.getLogger(__name__)


class Minkowski_Baseline_Model(BaseModel):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model, self).__init__(option)
        self._weight_classes = dataset.weight_classes
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.feature_dimension, D=option.D, **option.get("extra_options", {})
        )
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):

        # self.batch_idx = data.batch.squeeze()
        # coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        # self.input = ME.SparseTensor(features=data.x, coordinates=coords, device=device)
        # self.labels = data.y.to(device)

        voxel_size = 0.02

        torch.cuda.empty_cache()
        coords = ME.utils.batched_coordinates([data.coords / voxel_size], device=self.device)
        features = data.x.to(device=self.device)
        torch.cuda.empty_cache()

        self.input = ME.SparseTensor(
            features=features,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
            device=self.device)
        self.labels = data.y.to(self.device)

        # self.input = data.x.to(device), data.coords.to(device)
        # self.labels = data.y.to(device)

        # voxel_size = 0.1
        # coords = torch.cat([
        #     data.batch.unsqueeze(-1).int(), 
        #     (data.coords/voxel_size).int()], -1)
        # discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
        #     coordinates=coords,
        #     features=data.x,
        #     labels=data.y.view(-1, 1).int()
        # )
        # self.input = ME.SparseTensor(features=unique_feats, coordinates=discrete_coords, device=device)
        # self.labels = unique_labels.to(device)

    def forward(self, *args, **kwargs):
        # x, coords = self.input
        # voxel_size = 0.12
        # in_field = ME.TensorField(
        #     features = x,
        #     coordinates = ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.int32),
        #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        #     minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
        #     device=self.device
        # )

        # torch.cuda.empty_cache()

        # sinput = in_field.sparse()

        # torch.cuda.empty_cache()

        # soutput = self.model(sinput)

        # torch.cuda.empty_cache()

        # out_field = soutput.slice(in_field)
        # self.output = out_field.features
        # torch.cuda.empty_cache()

        torch.cuda.empty_cache()

        self.output = self.model(self.input).slice(self.input).F

        torch.cuda.empty_cache()

        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.device)
        if self.labels is not None:
            self.loss_seg = F.binary_cross_entropy_with_logits(self.output, self.labels)

        torch.cuda.empty_cache()

        # self.output = self.model(self.input)
        # if self._weight_classes is not None:
        #     self._weight_classes = self._weight_classes.to(self.device)
        # if self.labels is not None:
        #     # self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL, weight=self._weight_classes)
            
        #     self.labels = self.labels.view(*self.output.features.shape).float()

        #     weights = torch.ones(len(self.labels), 1, device=self.device)
        #     weights[self.labels == -100] = 0.0
        #     self.loss_seg = F.binary_cross_entropy_with_logits(self.output.features, self.labels, weight=weights)
        # return out_field.features.cpu()
        

    def backward(self):
        self.loss_seg.backward()
