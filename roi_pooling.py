import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.model.roi_layers import ROIAlign

import torch.autograd as ag
from torch.autograd.function import Function
from torch._thnn import type2backend


# class LevelMapper(object):
#     """Determine which FPN level each RoI in a set of RoIs should map to based
#     on the heuristic in the FPN paper.
#     """

#     def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
#         """
#         Arguments:
#             k_min (int)
#             k_max (int)
#             canonical_scale (int)
#             canonical_level (int)
#             eps (float)
#         """
#         self.k_min = k_min
#         self.k_max = k_max
#         self.s0 = canonical_scale
#         self.lvl0 = canonical_level
#         self.eps = eps

#     def __call__(self, boxes):

#         # Compute level ids
#         # Assign each ROI to a level in the pyramid based on the ROI area.
#         x1, y1, x2, y2 = boxes.chunk(4, dim=2)
#         h = y2 - y1
#         w = x2 - x1

#         # # Eqn.(1) in FPN paper
#         target_lvls = torch.floor(self.lvl0 + torch.log2(torch.sqrt(h*w) / self.s0 + self.eps))
#         target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
#         return target_lvls.to(torch.int64) - self.k_min


# class Pooler(nn.Module):
#     """
#     Pooler for Detection with or without FPN.
#     It currently hard-code ROIAlign in the implementation,
#     but that can be made more generic later on.
#     Also, the requirement of passing the scales is not strictly necessary, as they
#     can be inferred from the size of the feature map / size of original image,
#     which is available thanks to the BoxList.
#     """

#     def __init__(self, output_size, scales, sampling_ratio, canonical_level=4):
#         """
#         Arguments:
#             output_size (list[tuple[int]] or list[int]): output size for the pooled region
#             scales (list[float]): scales for each Pooler
#             sampling_ratio (int): sampling ratio for ROIAlign
#         """
#         super(Pooler, self).__init__()
#         poolers = []
#         for scale in scales:
#             poolers.append(
#                 ROIAlign(
#                     output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
#                 )
#             )
#         self.poolers = nn.ModuleList(poolers)
#         self.output_size = output_size
#         # get the levels in the feature map by leveraging the fact that the network always
#         # downsamples by a factor of 2 at each level.
#         lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
#         lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
#         self.map_levels = LevelMapper(
#             lvl_min, lvl_max, canonical_level=canonical_level
#         )

#     def convert_to_roi_format(self, boxes):
#         boxes = boxes[0,:,:]
#         ind = torch.zeros(boxes.shape[0], 1).cuda()
#         rois = torch.cat([ind, boxes], dim=1)
#         return rois

#     def forward(self, x, boxes):
#         """
#         Arguments:
#             x (list[Tensor]): feature maps for each level
#             boxes (list[BoxList]): boxes to be used to perform the pooling operation.
#         Returns:
#             result (Tensor)
#         """
#         num_levels = len(self.poolers)
#         rois = self.convert_to_roi_format(boxes)
#         if num_levels == 1:
#             return self.poolers[0](x[0], rois)

#         levels = self.map_levels(boxes)

#         num_rois = len(rois)
#         num_channels = x[0].shape[1]
#         output_size = self.output_size[0]

#         dtype, device = x[0].dtype, x[0].device
#         result = torch.zeros(
#             (num_rois, num_channels, output_size, output_size),
#             dtype=dtype,
#             device=device,
#         )
#         for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
#             idx_in_level = torch.nonzero(levels == level).squeeze(1)
#             rois_per_level = rois[idx_in_level]
#             result[idx_in_level] = pooler(per_level_feature, rois_per_level)

#         return result



class AdaptiveMaxPool2d(Function):

    def __init__(self, out_w, out_h):
        super(AdaptiveMaxPool2d, self).__init__()
        self.out_w = out_w
        self.out_h = out_h

    def forward(self, input):
        output = input.new()
        indices = input.new().long()
        self.save_for_backward(input)
        self.indices = indices
        self._backend = type2backend[input.type()]
        self._backend.SpatialAdaptiveMaxPooling_updateOutput(
                self._backend.library_state, input, output, indices,
                self.out_w, self.out_h)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        indices = self.indices
        grad_input = grad_output.new()
        self._backend.SpatialAdaptiveMaxPooling_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                indices)
        return grad_input, None

def adaptive_max_pool(input, size):
    return AdaptiveMaxPool2d(size[0],size[1])(input)

def roi_pooling(input, rois, size=(7,7), spatial_scale=1.0):
    assert(rois.dim() == 2)
    assert(rois.size(1) == 5)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:,1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        output.append(adaptive_max_pool(im, size))

    return torch.cat(output, 0)

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]
    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)]
    boxes = inputs[0]

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    x1, y1, x2, y2 = boxes.chunk(4, dim=2)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    roi_level = torch.floor(4 + torch.log2(torch.sqrt(h*w) /224 + 1e-6)) + 2
    roi_level = torch.clamp(roi_level, min=3, max=5)
    
    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    scales = [1.0/32, 1.0/64, 1.0/128]
    for i, level in enumerate(range(3, 6)):
        ix  = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:,0]
        level_boxes = boxes[ix.data, 0, :]
        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = torch.zeros(level_boxes.size()[0], 1).cuda()
        # feature_maps[i] = feature_maps[i].unsqueeze(0)  #CropAndResizeFunction needs batch dimension
        level_boxes = torch.cat([ind, level_boxes], dim=1)
        # level_boxes = torch.index_select(level_boxes, 1, torch.LongTensor([4,0,1,2,3]).cuda())
        pooled_features = roi_pooling(feature_maps[i], level_boxes, size=(pool_size,pool_size), spatial_scale=scales[i])
        # pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled
