import torch
from torch import nn
import torch.nn.functional as F

from .matcher import HungarianMatcher
from ..utils import (nested_tensor_from_tensor_list, interpolate, 
                        is_dist_avail_and_initialized, get_world_size,
                                box_cxcywh_to_xyxy, generalized_box_iou, box_iou)

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes:int, matcher: HungarianMatcher, weight_dict: dict, eos_coef: float, box_format = 'xyxy',
                                        focal_loss = True, focal_gamma = 2.0, focal_alpha = 0.3, 
                                                                                    losses = None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_loss = focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.box_format = box_format
        self.losses = list(self.get_loss_map()) if losses is None else losses
        self.eos_coef = eos_coef
    
    def loss_objectness(self, outputs, targets, indices, num_boxes):
        objectness_logits = outputs['objectness_logits'].squeeze(-1)
        
        idx = self._get_src_permutation_idx(indices)
        target = torch.zeros(objectness_logits.shape,
                                    dtype=objectness_logits.dtype, device=objectness_logits.device)
        target[idx] = 1.0
        
        loss_obj = sigmoid_focal_loss(objectness_logits, target, num_boxes, alpha = self.focal_alpha, gamma = self.focal_gamma)
        return {"loss_obj": loss_obj}
    
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        if "logits" not in outputs:
            raise KeyError("No predicted logits found in outputs")
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([_target["boxes"][i] for _target, (_, i) in zip(targets, indices)], dim=0)

        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["logits"]
        num_classes = src_logits.shape[-1]
        target_classes_original = torch.cat([_target["labels"][i] for _target, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], num_classes-1, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_original
        target = F.one_hot(target_classes, num_classes=num_classes)

        target_score_original = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_original[idx] = ious.to(target_score_original.dtype)
        target_score = target_score_original.unsqueeze(-1) * target

        loss = sigmoid_focal_loss(src_logits, target_score, num_boxes, alpha = self.focal_alpha, gamma = self.focal_gamma)
        return {"loss_vfl": loss}
    
    def loss_labels(self, outputs, targets, indices, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'logits' in outputs
        src_logits = outputs['logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        empty_weight = torch.ones(src_logits.shape[-1], device=src_logits.device)
        empty_weight[-1] = self.eos_coef

        target_classes = torch.full(src_logits.shape[:2], -100,
                                    dtype=torch.int64, device=src_logits.device)
        rand_batch_idx = torch.randint(0, src_logits.shape[0], (idx[0].shape[0], ))
        rand_query_idx = torch.randint(0, src_logits.shape[1], (idx[0].shape[0], ))
        rand_idx = (rand_batch_idx, rand_query_idx)
        target_classes[rand_idx] = src_logits.shape[-1]-1
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)

        losses = {'loss_ce': loss_ce}

        return losses
    

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs.logits
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs.pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    #TODO
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs.objectness_logits
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss_map(self):
        loss_map = {
            # 'labels': self.loss_labels,
            'label_vfl': self.loss_labels_vfl,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            "objectness": self.loss_objectness
            # 'masks': self.loss_masks
        }
        return loss_map
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = self.get_loss_map()
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses