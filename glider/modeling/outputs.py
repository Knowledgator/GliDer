from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling, ModelOutput

@dataclass
class GliDerOutput(ModelOutput):
    """
    Output type of [`Owlv2ForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        objectness_logits (`torch.FloatTensor` of shape `(batch_size, num_patches, 1)`):
            The objectness logits of all image patches. OWL-ViT represents images as a set of image patches where the
            total number of patches is (image_size / patch_size)**2.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`Owlv2TextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes image
            embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    objectness_logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
    
@dataclass
# Copied from transformers.models.owlvit.modeling_owlvit.OwlViTImageGuidedObjectDetectionOutput with OwlViT->Owlv2,OWL-ViT->OWLv2
class GliDerImageGuidedObjectDetectionOutput(ModelOutput):
    """
    Output type of [`Owlv2ForObjectDetection.image_guided_detection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual target image in the batch
            (disregarding possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual query image in the batch
            (disregarding possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes
            image embeddings for each patch.
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """

    logits: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    query_image_embeds: torch.FloatTensor = None
    target_pred_boxes: torch.FloatTensor = None
    query_pred_boxes: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

