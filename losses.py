import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils

def calc_iou(a, b):
    '''
    parameters: a, b = boxes 1 (anchor, prediction) and box 2 (annotation, ground-truth)
    returns intersection-over-union (IoU) of the two bounding boxes
    '''
    #area of predicted box
    area_a = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
    # area of ground-truth = delta x * delta y
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) 

    # calculate intersection width and height by finding max of x1, y1 and min of x2, y2
    # then find area using max(0, x2-x1+1)*max(0, y2-y1+1)
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], dim=1), b[:, 0]) 
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], dim=1), b[:, 1]) 

    iw = torch.clamp(iw, min=0) # clamps everything less than 0 to 0
    ih = torch.clamp(ih, min=0)

    intersection = iw * ih  # overlap

    # union area = areas of two boxes - intersection area
    ua = area_a + area_b - intersection
    ua = torch.clamp(ua, min=1e-8) 

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]
            
            good_anchors = anchors[0, positive_indices, :] 
            labels = torch.zeros(good_anchors.shape[0], 1).cuda()
            labels[:, 0] = assigned_annotations[positive_indices, 5].long()
            good_regression = regression[positive_indices, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return labels, good_anchors, good_regression, torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
class FocalLossModified(nn.Module):
    def __init__(self):
        super(FocalLossModified, self).__init__()
        self.cropBoxes = utils.CropBoxes()

    def forward(self, classifications, regressions, anchors, inputs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        img_batch, annotations, pairs = inputs
        cropped_imgs, stacked_pairs, labels = [None, None, None]

        # anchor box values: w, h, c_x, c_y
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            # label = labels[j, :, :]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            # similarity_annot = [bbox_annotation[:, 4]==label[:, :]]
            # classification_losses = F.binary_cross_entropy_with_logits(classification, similarity_annot)

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4) # set min and max of classification outputs

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            # find maximum IoU and its index for each anchor
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1  # matrix of -1's
            targets = targets.cuda()

            # anchors are assigned to ground-truth object boxes using an IoU threshold of 0.5
            # and to background if their IoU is in [0, 0.4).
            targets[torch.lt(IoU_max, 0.4), :] = 0  # if IoU_max < 0.4, set all values of the row to 0 
            positive_indices = torch.ge(IoU_max, 0.5)  # if IoU_max >= 0.5, set positive_indices to 1, otherwise 0

            num_positive_anchors = positive_indices.sum()  # compute number of 1's in positive_indices

            # take the rows of ground-truth coordinates with highest IoU. 76725 x 6
            assigned_annotations = bbox_annotation[IoU_argmax, :]  

            # obtain cropped images that have IoU > 0.5
            good_anchors = anchors[0, positive_indices, :] 
            good_labels = torch.zeros(good_anchors.shape[0], 1)
            good_labels[:, 0] = assigned_annotations[positive_indices, 5].long()
            good_imgs = self.cropBoxes(img_batch[j, :, :, :], good_anchors)
            good_pairs = pairs[j, :, :, :].unsqueeze(0).repeat(good_imgs.shape[0], 1, 1, 1)
            if cropped_imgs is None:
                cropped_imgs = good_imgs
                labels = good_labels
                stacked_pairs = good_pairs
            else:
                cropped_imgs = torch.cat((cropped_imgs, good_imgs))
                labels = torch.cat((labels, good_labels))
                stacked_pairs = torch.cat((stacked_pairs, good_pairs))

            targets[positive_indices, :] = 0 # if IoU_max >= 0.5, set targets = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1 # set each anchor box to one label

            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            # if targets == 1, alpha_factor = alpha, else, alpha = 1 - alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            # if targets == 1, use 1 - classification probability
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            # binary cross entropy
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            # if positive_indices.sum() > 0:
            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                # anchor box values: w, h, c_x, c_y
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # ground truth values: w, h, c_x, c_y
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                # use formula (2) in Faster RCNN paper 
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                # negative_indices = 1 - positive_indices
                # L1 loss: 
                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        # return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
        return cropped_imgs, stacked_pairs, labels, torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
