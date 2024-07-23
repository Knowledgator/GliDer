from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod
from torch import nn
import torch

from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers import PreTrainedModel, AutoModel, AutoConfig

from ..config import GliDerConfig
from .outputs import GliDerImageGuidedObjectDetectionOutput, GliDerOutput
from .layers import (Vision2TextProjector, Text2VisionProjector, create_projection_layer, 
                                                                CrossFuser, LayersFusion, Decoder)
from ..owl.config import Owlv2TextConfig, Owlv2VisionConfig
from ..owl.model import Owlv2TextModel
from .vitdet import VitDetModel, VitDetConfig
from .heads import ClassPredictionHead, BoxPredictionHead
from .towers import CLIPVisionModel, Owlv2VisionModel
from ..utils import box_iou, generalized_box_iou
    
class GliDerPreTrainedModel(PreTrainedModel):
    config_class = GliDerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]

    def _init_weights(self, module):
        std = (
            self.config.initializer_factor
            if hasattr(self.config, "initializer_range")
            else self.config.initializer_factor
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class BaseDetector(ABC, GliDerPreTrainedModel):
    """
    Base class for GliDer object detection models.
    
    This class implements common functionality for object detection models
    using a vision encoder
    """
    config_class = GliDerConfig

    def __init__(self, config: GliDerConfig, vision_config=None, from_pretrained=False,
                                                            image_size=None, patch_size=None):
        super().__init__(config)
        """
        Initialize the BaseGliDerObjectDetection model.

        Args:
            config (GliDerConfig): The model configuration.
            vision_config: Configuration for the vision model.
            from_pretrained (bool): Whether to load pre-trained weights.
            image_size (int, optional): Size of the input image.
            patch_size (int, optional): Size of image patches.
        """

        if vision_config is None and config.vision_config is None:
            if 'owl' in config.vision_model_name:
                vision_config = Owlv2VisionConfig.from_pretrained(config.vision_model_name)

            else:
                vision_config = AutoConfig.from_pretrained(config.vision_model_name)
                if 'clip' in config.vision_model_name or 'owl' in config.vision_model_name:
                    vision_config = vision_config.vision_config
            if config.is_vit_det:
                vision_config = VitDetConfig(**vision_config.to_dict(),image_size=image_size)
            config.vision_config = vision_config
        
        if not hasattr(config.vision_config, 'hidden_size'):
            config.vision_config.hidden_size = config.vision_config.d_model #for detr models
        
        if hasattr(config.vision_config, 'image_size') and image_size!=None:
            config.vision_config.image_size = image_size
        
        if hasattr(config.vision_config, 'patch_size') and patch_size!=None:
            config.vision_config.patch_size = patch_size
        
        self.has_cls_token=False
        if hasattr(config.vision_config, "architectures"):
            if 'clip' in config.vision_model_name or 'owl' in config.vision_model_name:
                self.has_cls_token=True
            elif config.vision_config.architectures is None:
                pass
            elif ('classification' in config.vision_config.architectures[0].lower()
                    and 'swin' not in config.vision_model_name):
                self.has_cls_token=True
        
        self.vision_model = self.init_vision_tower(from_pretrained)

        if config.vision_layers_fusion:
            self.vision_layers_fuser = LayersFusion(config.vision_config.num_hidden_layers, config.vision_config.hidden_size)

        self.box_head = BoxPredictionHead(config)
        self.objectness_head = BoxPredictionHead(config, out_dim=1)

        if not hasattr(config.vision_config, 'layer_norm_eps'):
            config.vision_config.layer_norm_eps = 1e-6
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        
        self.sigmoid = nn.Sigmoid()

        if not hasattr(config.vision_config, 'image_size'):
            self.sqrt_num_patches = int(config.vision_config.num_queries**0.5)
        else:
            self.sqrt_num_patches = config.vision_config.image_size // (config.vision_config.patch_size)
        
        if config.vision_config.is_encoder_decoder or config.decoder_layers:
            if config.decoder_layers:
                num_queries=config.num_queries
            else:
                num_queries = config.vision_config.num_queries
            
            if config.has_box_bias:
                self.box_bias = self.compute_box_bias_encoder_decoder(num_queries)

        elif config.has_box_bias:
            self.box_bias = self.compute_box_bias(self.sqrt_num_patches)
        else:
            self.box_bias=None

        if config.decoder_layers:
            self.decoder = Decoder(config)

    def init_vision_tower(self, from_pretrained):
        """
        Initialize the vision tower of the model.

        Args:
            from_pretrained (bool): Whether to load pre-trained weights.

        Returns:
            The initialized vision model.
        """
        if from_pretrained:
            if 'clip' in self.config.vision_model_name:
                tower = CLIPVisionModel.from_pretrained(self.config.vision_model_name)
            elif 'owl' in self.config.vision_model_name:
                tower = Owlv2VisionModel.from_pretrained(self.config.vision_model_name)
            elif 'vitdet' in self.config.vision_model_name:
                tower = VitDetModel.from_pretrained(self.config.vision_model_name)
            else:
                tower = AutoModel.from_pretrained(self.config.vision_model_name, attn_implementation='eager',
                                                                image_size=self.config.vision_config.image_size,
                                                                patch_size=self.config.vision_config.patch_size,
                                                                                    ignore_mismatched_sizes=True)
        else:
            if 'clip' in self.config.vision_model_name:
                tower = CLIPVisionModel(self.config.vision_config)
            elif 'owl' in self.config.vision_model_name:
                tower = Owlv2VisionModel(self.config.vision_config)
            elif 'vitdet' in self.config.vision_model_name:
                tower = VitDetModel(self.config.vision_config)
            else:
                tower = AutoModel.from_config(self.config.vision_config)
        return tower
    
    @staticmethod
    def normalize_grid_corner_coordinates(num_patches: int) -> torch.Tensor:
        """
        Compute normalized grid corner coordinates.

        Args:
            num_patches (int): Number of patches along each dimension.

        Returns:
            torch.Tensor: Normalized grid corner coordinates.
        """
        # Create grid coordinates using torch
        x_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        y_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")

        # Stack the coordinates and divide by num_patches
        box_coordinates = torch.stack((xx, yy), dim=-1)
        box_coordinates /= num_patches

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.view(-1, 2)

        return box_coordinates

    def objectness_predictor(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """
        Predict the probability that each image feature token is an object.

        Args:
            image_features (torch.FloatTensor): Features extracted from the image.

        Returns:
            torch.FloatTensor: Objectness scores.
        """
        image_features = image_features.detach()
        objectness_logits = self.objectness_head(image_features)
        objectness_logits = objectness_logits[..., 0]
        return objectness_logits

    def compute_box_bias(self, num_patches: int, feature_map: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """
        Compute box bias for the model.

        Args:
            num_patches (int): Number of patches along each dimension.
            feature_map (Optional[torch.FloatTensor]): Deprecated parameter.

        Returns:
            torch.Tensor: Computed box bias.
        """
        if feature_map is not None:
            raise ValueError("feature_map has been deprecated as an input. Please pass in num_patches instead")
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(num_patches)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def compute_box_bias_encoder_decoder(self, num_queries: int) -> torch.Tensor:
        """
        Compute box bias for encoder-decoder models.

        Args:
            num_queries (int): Number of queries.

        Returns:
            torch.Tensor: Computed box bias for encoder-decoder models.
        """
        # Compute bias for decoder queries
        query_positions = torch.linspace(0, 1, num_queries + 2)[1:-1].unsqueeze(-1).repeat(1, 2)
        box_coord_bias = torch.log(query_positions + 1e-4) - torch.log1p(-query_positions + 1e-4)
        
        # The box size is biased to a default size (e.g., 1/sqrt(num_queries))
        default_size = 1.0 / (num_queries ** 0.5)
        box_size = torch.full_like(box_coord_bias, default_size)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)
        
        # Combine coordinate and size biases
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        
        return box_bias

    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Predict bounding boxes from image features.

        Args:
            image_feats (torch.FloatTensor): Features extracted from the image.

        Returns:
            torch.FloatTensor: Predicted bounding boxes.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        if self.box_bias is not None:
            box_bias = self.box_bias.to(image_feats.device)
            pred_boxes += box_bias
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor]:
        """
        Embed input images into feature representations.

        Args:
            pixel_values (torch.FloatTensor): Input image pixel values.

        Returns:
            Tuple[torch.FloatTensor]: Image embeddings and vision model outputs.
        """
        # Get Owlv2Model vision embeddings (same as CLIP)
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)

        # Apply post_layernorm to last_hidden_state, return non-projected output
        last_hidden_state = vision_outputs[0]
        image_embeds = self.layer_norm(last_hidden_state)

        if self.has_cls_token:
            image_embeds = image_embeds[:,1:,:]

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return (image_embeds, vision_outputs)

    def text_embedder(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Embed input text into feature representations.

        Args:
            input_ids (torch.LongTensor): Input text token ids.
            attention_mask (Optional[torch.LongTensor]): Attention mask for input text.

        Returns:
            Tuple[torch.FloatTensor]: Text embeddings and text model outputs.
        """
        # Get text embeddings from the text model
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Get the last hidden state
        last_hidden_state = text_outputs[0]

        return (last_hidden_state, text_outputs)

    @abstractmethod
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Predict object classes from image features.

        Args:
            image_feats (torch.FloatTensor): Image features.
            query_embeds (Optional[torch.FloatTensor]): Query embeddings.
            query_mask (Optional[torch.Tensor]): Query mask.

        Returns:
            Tuple[torch.FloatTensor]: Predicted logits and class embeddings.
        """
        pass

    def image_text_embedder(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Embed both image and text inputs.

        Args:
            input_ids (torch.Tensor): Input text token ids.
            pixel_values (torch.FloatTensor): Input image pixel values.
            attention_mask (torch.Tensor): Attention mask for input text.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.

        Returns:
            Tuple[torch.FloatTensor]: Combined image and text embeddings.
        """
        pass
 
    def embed_image_query(
        self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Embed image query features.

        Args:
            query_image_features (torch.FloatTensor): Query image features.
            query_feature_map (torch.FloatTensor): Query feature map.

        Returns:
            torch.FloatTensor: Embedded image query.
        """
        pass
 
    def image_guided_detection(
        self,
        pixel_values: torch.FloatTensor,
        query_pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> GliDerImageGuidedObjectDetectionOutput:
        """
        Perform image-guided object detection.

        Args:
            pixel_values (torch.FloatTensor): Input image pixel values.
            query_pixel_values (Optional[torch.FloatTensor]): Query image pixel values.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary or tuple.

        Returns:
            GliDerImageGuidedObjectDetectionOutput: Detection results.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Compute feature maps for the input and query images
        query_feature_map = self.image_embedder(pixel_values=query_pixel_values)[0]
        image_feats, vision_outputs = self.image_embedder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
        query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        # Get top class embedding and best box index for each query image in batch
        query_embeds, best_box_indices, query_pred_boxes = self.embed_image_query(query_image_feats, query_feature_map)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats=image_feats, query_embeds=query_embeds)

        # Predict object boxes
        target_pred_boxes = self.box_predictor(image_feats)

        if not return_dict:
            output = (
                image_feats,
                query_feature_map,
                target_pred_boxes,
                query_pred_boxes,
                pred_logits,
                class_embeds,
                vision_outputs.to_tuple(),
            )
            output = tuple(x for x in output if x is not None)
            return output

        return GliDerImageGuidedObjectDetectionOutput(
            image_embeds=image_feats,
            query_image_embeds=query_feature_map,
            target_pred_boxes=target_pred_boxes,
            query_pred_boxes=query_pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            text_model_output=None,
            vision_model_output=vision_outputs,
        )


class BaseGliDer(BaseDetector):
    """
    Base class for zero-shot GliDer object detection models.
    
    This class implements common functionality for object detection models
    using a vision encoder and text encoder.
    """
    def __init__(self, config: GliDerConfig, text_config=None, vision_config=None, 
                                            from_pretrained=False, image_size=None, patch_size=None):
        super().__init__(config, vision_config, from_pretrained, image_size, patch_size)
        """
        Initialize the BaseGliDer model.

        Args:
            config (GliDerConfig): The model configuration.
            text_config: Configuration for the text model.
            vision_config: Configuration for the vision model.
            from_pretrained (bool): Whether to load pre-trained weights.
            image_size (int, optional): Size of the input image.
            patch_size (int, optional): Size of image patches.
        """
        if text_config is None and config.text_config is None:
            if 'owl' in config.text_model_name:
                text_config = Owlv2TextConfig.from_pretrained(config.text_model_name)
            else:
                text_config = AutoConfig.from_pretrained(config.text_model_name)
                if 'clip' in config.text_model_name:
                    text_config = text_config.text_config
            config.text_config = text_config

        self.vision_text_projector = Vision2TextProjector(config)

        self.text_vision_projector = Text2VisionProjector(config)

        self.text_model = self.init_text_tower(from_pretrained)

        self.class_head = ClassPredictionHead(config)

        if config.deep_fusion:
            self.fuser = self.init_vision_tower(from_pretrained)
            self.fusion_dropout = torch.nn.Dropout(p=config.fusion_dropout, inplace=False)

        if config.post_fusion_schema:
            self.cross_fuser = CrossFuser(config.vision_config.hidden_size,
                                          config.text_config.hidden_size,
                                          num_heads=config.vision_config.num_attention_heads,
                                            num_layers=1,
                                            dropout=config.fusion_dropout, 
                                            schema=config.post_fusion_schema)

        if config.text_layers_fustion:
            self.text_layers_fuser = LayersFusion(config.text_config.num_hidden_layers, config.text_config.hidden_size)
  
    def fusing(self, text_features, vision_features):
        """
        Fuse text and vision features.

        Args:
            text_features (torch.Tensor): Text features.
            vision_features (torch.Tensor): Vision features.

        Returns:
            tuple: Updated text and vision features after fusion.
        """
        fused_features = torch.cat([text_features, vision_features], dim=1)
        fused_outputs = self.fuser(inputs_embeds=fused_features)
        updated_feautures = fused_outputs[0]
        updated_text_features = updated_feautures[:, :text_features.shape[1],:]
        updated_vision_features =  updated_feautures[:, text_features.shape[1]:,:]
        return (updated_text_features, updated_vision_features)
    
    def features_enhancement(self, text_embeds, image_embeds, text_mask=None, image_mask=None):
        """
        Enhance text and image features through cross-modal fusion.

        Args:
            text_embeds (torch.Tensor): Text embeddings.
            image_embeds (torch.Tensor): Image embeddings.
            text_mask (torch.Tensor, optional): Mask for text embeddings.
            image_mask (torch.Tensor, optional): Mask for image embeddings.

        Returns:
            tuple: Enhanced text and image embeddings.
        """
        text_embeds, image_embeds = self.cross_fuser(text_embeds, image_embeds, text_mask, image_mask)
        return text_embeds, image_embeds
    
    def init_text_tower(self, from_pretrained):
        """
        Initialize the text tower of the model.

        Args:
            from_pretrained (bool): Whether to load pre-trained weights.

        Returns:
            The initialized text model.
        """
        if from_pretrained:
            if 'clip' in self.config.text_model_name:
                tower = CLIPTextModel.from_pretrained(self.config.text_model_name)
            elif 'owl' in self.config.text_model_name:
                tower = Owlv2TextModel.from_pretrained(self.config.text_model_name)
            else:
                tower = AutoModel.from_pretrained(self.config.text_model_name)
        else:
            if 'clip' in self.config.text_model_name:
                tower = CLIPTextModel(self.config.text_config)
            elif 'owl' in self.config.text_model_name:
                tower = Owlv2TextModel(self.config.text_config)
            else:
                tower = AutoModel.from_config(self.config.text_config)
        return tower
    
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)
    

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> GliDerOutput:
        """
        Forward pass of the BaseGliDer model.

        Args:
            input_ids (torch.Tensor): Input text token ids.
            pixel_values (torch.FloatTensor): Input image pixel values.
            attention_mask (Optional[torch.Tensor]): Attention mask for input text.
            pixel_mask (Optional[torch.Tensor]): Mask for input image pixels.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary or tuple.

        Returns:
            GliDerOutput: Model outputs including predictions and embeddings.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Embed images and text queries
        query_embeds, image_feats, text_outputs, vision_outputs = self.image_text_embedder(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        batch_size = pixel_values.shape[0]
        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        max_text_queries = input_ids.shape[0] // batch_size
        # query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        if self.training and self.config.num_query_groups>1:
            input_ids = input_ids.repeat(self.config.num_query_groups, 1, 1)
        query_mask = input_ids[..., 0] > 0

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)

        # Predict objectness
        objectness_logits = self.objectness_predictor(image_feats)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats)

        if not return_dict:
            output = (
                pred_logits,
                objectness_logits,
                pred_boxes,
                query_embeds,
                class_embeds,
                text_outputs.to_tuple(),
                vision_outputs.to_tuple(),
            )
            output = tuple(x for x in output if x is not None)
            return output

        return GliDerOutput(
            image_embeds=image_feats,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            objectness_logits=objectness_logits,
            class_embeds=class_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
    
class GliDer(BaseGliDer):
    """
    GliDer (Generalist Language-Image Detection) model.
    
    This class extends BaseGliDer with specific implementations for
    image-text embedding and image query embedding.
    """
    def image_text_embedder(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Embed both image and text inputs.

        Args:
            input_ids (torch.Tensor): Input text token ids.
            pixel_values (torch.FloatTensor): Input image pixel values.
            attention_mask (torch.Tensor): Attention mask for input text.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.

        Returns:
            Tuple[torch.FloatTensor]: Text embeddings, image embeddings, text model outputs, and image model outputs.
        """
        # Encode text and image
        batch_size=pixel_values.shape[0]
        max_text_queries = input_ids.shape[0] // batch_size
        input_ids_ = input_ids.reshape(batch_size, max_text_queries, -1)[0]
        attention_mask_ = attention_mask.reshape(batch_size, max_text_queries, -1)[0]

        text_outputs = self.text_model(
            input_ids=input_ids_,
            attention_mask=attention_mask_,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        #text features
        text_embeds = text_outputs[1]
        text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        image_outputs = self.vision_model(pixel_values, output_attentions=output_attentions,
                                                                    output_hidden_states=True)
        #vision features
        if self.config.vision_config.is_encoder_decoder:
            layers_states = image_outputs.decoder_hidden_states
        else:
            layers_states = image_outputs.hidden_states
        if self.config.vision_layers_fusion: 
            image_embeds = self.vision_layers_fuser(layers_states[1:])
        elif self.config.num_query_groups>1 and self.training:
            image_embeds = torch.cat(layers_states[-self.config.num_query_groups:], dim=0)
            text_embeds = text_embeds.repeat(self.config.num_query_groups, 1, 1)
        else:
            image_embeds = image_outputs.last_hidden_state

        if self.config.is_vit_det:
            image_embeds = image_embeds.reshape(batch_size, image_embeds.shape[1], -1).transpose(1, 2)

        if self.has_cls_token and not self.config.is_vit_det:
            image_embeds = image_embeds[:,1:,:]

        # Get image embeddings
        if self.config.deep_fusion:
            text_embeds, image_embeds = self.fusing(text_embeds, image_embeds)

        if self.config.post_fusion_schema:
            text_embeds, image_embeds = self.features_enhancement(text_embeds, image_embeds)
            
        image_embeds = self.layer_norm(image_embeds)

        if self.config.decoder_layers:
            image_embeds = self.decoder(image_embeds)

        return (text_embeds, image_embeds, text_outputs, image_outputs)
    
    def embed_image_query(
        self, query_image_features: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Embed image query features and select the best matching embeddings.

        Args:
            query_image_features (torch.FloatTensor): Features extracted from query images.

        Returns:
            Tuple[torch.FloatTensor]: Best class embeddings, indices of best boxes, and predicted boxes.
        """
        _, class_embeds = self.class_predictor(query_image_features)
        pred_boxes = self.box_predictor(query_image_features)

        # Loop over query images
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes.device

        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # If there are no overlapping boxes, fall back to generalized IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # Use an adaptive threshold to include all boxes within 80% of the best IoU
            iou_threshold = torch.max(ious) * 0.8

            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)

        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None

        return query_embeds, box_indices, pred_boxes


class UniGliDer(BaseGliDer):
    """
    Uni-encoder text representation Generalist Language-Image Detector (UniGliDer) model.
    
    This class extends BaseGliDer with a unified approach to process
    both textual labels in a single forward pass.
    """
    def __init__(self, config: GliDerConfig, text_config=None, vision_config=None, 
                                    from_pretrained=False, image_size=None, patch_size=None):
        super().__init__(config, text_config, vision_config, from_pretrained, image_size, patch_size)
        self.image_projector = create_projection_layer(config.vision_hidden_size, dropout = 0.1)
        self.text_projector = create_projection_layer(config.vision_hidden_size, out_dim = config.text_hidden_size, dropout = 0.1)

    def _extract_prompt_features_and_word_embeddings(self, token_embeds, input_ids, attention_mask):
        """
        Extract prompt features and word embeddings from token embeddings.

        Args:
            token_embeds (torch.Tensor): Token embeddings.
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask for input tokens.

        Returns:
            tuple: Prompt embeddings and prompt embedding mask.
        """
        batch_size, sequence_length, embed_dim = token_embeds.shape

        # getting prompt embeddings
        class_token_mask = input_ids == self.config.class_token_index

        num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

        max_embed_dim = self.config.max_objects#num_class_tokens.max()
        aranged_class_idx = torch.arange(max_embed_dim, 
                                            dtype=attention_mask.dtype, 
                                            device=token_embeds.device).expand(batch_size, -1)
        
        batch_indices, target_class_idx = torch.where(aranged_class_idx<num_class_tokens)
        _, class_indices = torch.where(class_token_mask)
        # class_indices+=1

        prompts_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
        )
        prompts_embedding_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=token_embeds.device
        )

        prompts_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]
        prompts_embedding_mask[batch_indices, target_class_idx] = 1
    
        return prompts_embedding, prompts_embedding_mask
    
    def image_text_embedder(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Embed both image and text inputs in a unified manner.

        Args:
            input_ids (torch.Tensor): Input text token ids.
            pixel_values (torch.FloatTensor): Input image pixel values.
            attention_mask (torch.Tensor): Attention mask for input text.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.

        Returns:
            Tuple[torch.FloatTensor]: Label embeddings, image embeddings, text outputs, 
                                      vision outputs, and label embedding mask.
        """
        # Encode text and image
        text_embeds, text_outputs = self.text_embedder(input_ids = input_ids,
                                          attention_mask=attention_mask)
        
        batch_size, num_tokens = text_embeds.shape[:2]

        text_embeds = self.text_vision_projector(text_embeds)

        image_input_embeds = self.vision_model.vision_model.embeddings(pixel_values)
        inputs_embeds = torch.cat([text_embeds, image_input_embeds], dim=1)
        image_outputs = self.vision_model(inputs_embeds=inputs_embeds, output_attentions=output_attentions,
                                                    output_hidden_states=output_hidden_states)


        label_embeds = self.text_projector(image_outputs[0][:,:num_tokens,:])
        label_embeds, label_embedding_mask = self._extract_prompt_features_and_word_embeddings(label_embeds, input_ids, attention_mask)
        image_embeds = self.image_projector(image_outputs[0][:,num_tokens+1:,:])
        image_embeds = self.layer_norm(image_embeds)

        return (label_embeds, image_embeds, text_outputs, image_outputs, label_embedding_mask)
    
    
    def embed_image_query(
        self, query_image_features: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Embed image query features and select the best matching embeddings.

        Args:
            query_image_features (torch.FloatTensor): Features extracted from query images.

        Returns:
            Tuple[torch.FloatTensor]: Best class embeddings, indices of best boxes, and predicted boxes.
        """
        _, class_embeds = self.class_predictor(query_image_features)
        pred_boxes = self.box_predictor(query_image_features)

        # Loop over query images
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes.device

        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # If there are no overlapping boxes, fall back to generalized IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # Use an adaptive threshold to include all boxes within 80% of the best IoU
            iou_threshold = torch.max(ious) * 0.8

            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)

        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None

        return query_embeds, box_indices, pred_boxes

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> GliDer:
        """
        Forward pass of the UniGliDer model.

        Args:
            input_ids (torch.Tensor): Input text token ids.
            pixel_values (torch.FloatTensor): Input image pixel values.
            attention_mask (Optional[torch.Tensor]): Attention mask for input text.
            pixel_mask (Optional[torch.Tensor]): Mask for input image pixels.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary or tuple.

        Returns:
            GliDerOutput: Model outputs including predictions and embeddings.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Embed images and text queries
        query_embeds, image_feats, text_outputs, vision_outputs, query_mask = self.image_text_embedder(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        batch_size, _, _ = image_feats.shape

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)

        # Predict objectness
        objectness_logits = self.objectness_predictor(image_feats)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats)

        if not return_dict:
            output = (
                pred_logits,
                objectness_logits,
                pred_boxes,
                query_embeds,
                image_feats,
                class_embeds,
                text_outputs.to_tuple(),
                vision_outputs.to_tuple(),
            )
            output = tuple(x for x in output if x is not None)
            return output

        return GliDerOutput(
            image_embeds=image_feats,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            objectness_logits=objectness_logits,
            class_embeds=class_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class SupervisedGliDer(BaseDetector):
    """
    Supervised Object Detector.
    
    This class extends BaseDetector with a supervised approach for object detection,
    using a fixed number of object classes.
    """
    def __init__(self, config: GliDerConfig, num_labels = 91, vision_config=None, from_pretrained=False):
        super().__init__(config, vision_config, from_pretrained)
        """
        Initialize the SupervisedDecForObjectDetection model.

        Args:
            config (GliDerConfig): The model configuration.
            num_labels (int): Number of object classes to detect.
            vision_config: Configuration for the vision model.
            from_pretrained (bool): Whether to load pre-trained weights.
        """
        self.class_head = BoxPredictionHead(config, out_dim = num_labels)

    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
        """
        pred_logits = self.class_head(image_feats)

        return pred_logits
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> GliDerOutput:
        """
        Forward pass of the SupervisedDecForObjectDetection model.

        Args:
            pixel_values (torch.FloatTensor): Input image pixel values.
            pixel_mask (Optional[torch.Tensor]): Mask for input image pixels.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary or tuple.

        Returns:
            GliDerOutput: Model outputs including predictions and embeddings.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # Embed images and text queries
        feature_map, vision_outputs = self.image_embedder(
            pixel_values=pixel_values,
        )
        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        pred_logits = self.class_predictor(image_feats)
        # Predict objectness
        objectness_logits = self.objectness_predictor(image_feats)
        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats)

        if not return_dict:
            output = (
                pred_logits,
                objectness_logits,
                pred_boxes,
                feature_map,
                vision_outputs.to_tuple(),
            )
            output = tuple(x for x in output if x is not None)
            return output

        return GliDerOutput(
            image_embeds=feature_map,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            objectness_logits=objectness_logits,
            vision_model_output=vision_outputs,
        )