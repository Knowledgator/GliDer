import torch
from torch import nn
import torch.nn.functional as F

from ..utils import box_cxcywh_to_xyxy, generalized_box_iou

class DistilationCriterion(nn.Module):
    def __init__(self, temperature=1.5, weight_dict=None):
        super().__init__()
        self.temperature = temperature
        self.weight_dict = weight_dict if weight_dict is not None else {}

    def loss_classification(self, teacher_logits, student_logits):
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.temperature**2)
        return soft_targets_loss

    def loss_objectness(self, teacher_objectness, student_objectness):
        return F.binary_cross_entropy_with_logits(student_objectness, teacher_objectness.sigmoid())

    def loss_boxes(self, teacher_boxes, student_boxes):
        losses = {}
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(teacher_boxes),
            box_cxcywh_to_xyxy(student_boxes)))
        losses['giou'] = loss_giou.sum() / teacher_boxes.shape[1]
        losses['boxes'] = F.l1_loss(student_boxes, teacher_boxes)
        return losses

    def get_loss_map(self):
        loss_map = {
            'classification': self.loss_classification,
            "objectness": self.loss_objectness,
            "boxes": self.loss_boxes
        }
        return loss_map

    def forward(self, teacher_output, student_output):
        teacher_logits, teacher_boxes, teacher_objectness = teacher_output.logits, teacher_output.pred_boxes, teacher_output.objectness_logits
        student_logits, student_boxes, student_objectness = student_output.logits, student_output.pred_boxes, student_output.objectness_logits

        loss_map = self.get_loss_map()
        losses = {}

        for loss_name, loss_func in loss_map.items():
            if loss_name == 'classification':
                losses[loss_name] = loss_func(teacher_logits, student_logits)
            elif loss_name == 'objectness':
                losses[loss_name] = loss_func(teacher_objectness, student_objectness)
            elif loss_name == 'boxes':
                losses.update(loss_func(teacher_boxes, student_boxes))

        # Apply weights to losses
        weighted_losses = {k: self.weight_dict.get(k, 1.0) * v for k, v in losses.items()}

        # Sum up all losses
        total_loss = sum(weighted_losses.values())

        return total_loss