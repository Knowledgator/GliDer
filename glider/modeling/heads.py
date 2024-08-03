from typing import Optional, Union, Tuple

import torch
from torch import nn

# from ..config import GliDerConfig

class ClassPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        out_dim = config.text_hidden_size
        self.query_dim = config.vision_hidden_size

        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        if config.zero_class_embedding:
            self.zero_class_embedding = nn.Parameter(torch.randn(1, out_dim))
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # if self.config.zero_class_embedding:
        #     query_embeds = torch.cat([query_embeds, self.zero_class_embedding.unsqueeze(0)
        #                                         .expand(query_embeds.size(0), 1, -1)], dim=1)
        # Get class predictions
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        # if self.config.zero_class_embedding:
        #     zero_class_mask = torch.ones((query_mask.shape[0], 1), dtype=query_mask.dtype, device = query_mask.device)
        #     query_mask = torch.cat([query_mask, zero_class_mask], dim=-1)
        
        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)

            pred_logits = pred_logits.to(torch.float64)
            pred_logits = torch.where(query_mask == 0, -1e6, pred_logits)
            pred_logits = pred_logits.to(torch.float32)

        return (pred_logits, image_class_embeds)

class BoxPredictionHead(nn.Module):
    def __init__(self, config, out_dim: int = 4):
        super().__init__()

        width = config.vision_hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output
