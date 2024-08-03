from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

# from ..config import GliDerConfig

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout) -> None:
        super().__init__()
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.attention_head_size=hidden_size//num_heads
        self.attention_probs_dropout_prob=dropout
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key=None, value=None, head_mask=None, attn_mask=None):
        query = self.transpose_for_scores(self.query_layer(query))
        if key is None:
            key = self.transpose_for_scores(self.key_layer(query))
        else:
            key = self.transpose_for_scores(self.key_layer(key))
        if value is None and key is None:
            value = self.transpose_for_scores(self.value_layer(query))
        elif value is None and key is not None:
            value = self.transpose_for_scores(self.value_layer(key))
        else:
            value = self.transpose_for_scores(self.value_layer(value))

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None
    
class Vision2TextProjector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_hidden_size, config.text_hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_hidden_size, config.text_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Text2VisionProjector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_1 = nn.Linear(config.text_hidden_size, config.vision_hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.vision_hidden_size, config.vision_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def create_projection_layer(hidden_size: int, dropout: float, out_dim: int = None) -> nn.Sequential:
    """
    Creates a projection layer with specified configurations.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim)
    )

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output, _ = self.self_attn(q, k, v, attn_mask=mask)
        return self.norm(x + self.dropout(attn_output))

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value=None, mask=None):
        if value is None:
            value = self.v_proj(key)
        attn_output, _ = self.cross_attn(query, key, value, attn_mask=mask)
        return self.norm(query + self.dropout(attn_output))

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.vision_config.hidden_size

        self.self_attn = SelfAttentionBlock(
            embed_dim=self.embed_dim,
            num_heads=config.vision_config.num_attention_heads,
            dropout=config.vision_config.hidden_dropout_prob,
        )
        self.dropout = config.vision_config.hidden_dropout_prob
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.encoder_attn = CrossAttentionBlock(
            embed_dim=self.embed_dim,
            num_heads=config.vision_config.num_attention_heads,
            dropout=config.vision_config.hidden_dropout_prob,
        )
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
        )

        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                query=hidden_states,
                key=encoder_hidden_states,
                mask=encoder_attention_mask,
            )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.vision_config.hidden_size)
        self.query_embeddings = nn.Embedding(config.num_queries, config.vision_config.hidden_size)

    def forward(
        self,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = encoder_hidden_states.shape[0]
        hidden_states = self.query_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states)
        
        return hidden_states #batch_size, num_queries, hidden_size
    
class CrossFuser(nn.Module):
    def __init__(self, d_model, query_dim, num_heads=8, num_layers=1, dropout=0.1, schema='l2l-l2i'):
        super().__init__()
        self.d_model = d_model
        self.schema = schema.split('-')
        layers = []
        for _ in range(num_layers):
            layer = []
            for attn_type in self.schema:
                if attn_type in {'l2l', 'i2i'}:
                    layer.append(SelfAttentionBlock(d_model, num_heads, dropout))
                else:
                    layer.append(CrossAttentionBlock(d_model, num_heads, dropout))
            layer = nn.ModuleList(layer)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.dense_i = nn.Linear(query_dim, d_model)
        self.dense_o = nn.Linear(d_model, query_dim)

        
    def forward(self, query, key, query_mask=None, key_mask=None):
        query = self.dense_i(query)

        for sublayers in self.layers:
            for id, layer in enumerate(sublayers):
                if self.schema[id] == 'l2l':
                    if query_mask is not None:
                        self_attn_mask = query_mask.unsqueeze(1) * query_mask.unsqueeze(2)
                    else:
                        self_attn_mask = None
                    query = layer(query, mask=self_attn_mask)
                elif self.schema[id] == 'i2i':
                    if key_mask is not None:
                        self_attn_mask = key_mask.unsqueeze(1) * key_mask.unsqueeze(2)
                    else:
                        self_attn_mask = None
                    key = layer(key, mask=self_attn_mask)
                elif self.schema[id] == 'l2i':
                    if query_mask is not None and key_mask is not None:
                        cross_attn_mask = query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)
                    else:
                        cross_attn_mask = None
                    query = layer(query, key, mask=cross_attn_mask)
                elif self.schema[id] == 'i2l':
                    if query_mask is not None and key_mask is not None:
                        cross_attn_mask = key_mask.unsqueeze(-1) * query_mask.unsqueeze(1)
                    else:
                        cross_attn_mask = None
                    key = layer(key, query, mask=cross_attn_mask)
        query=self.dense_o(query)   
        return query, key

class LayersFusion(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        
        # Squeeze operation
        self.squeeze = nn.Linear(hidden_size, 1)
        
        # Excitation operation
        self.W1 = nn.Linear(num_layers, num_layers // 2)
        self.W2 = nn.Linear(num_layers // 2, num_layers)
        
        # Final projection
        self.output_projection = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, encoder_outputs):
        # encoder_outputs is a list of tensors, each of shape [B, L, D]
        B, L, D = encoder_outputs[0].shape
        
        # Concatenate all layers
        U = torch.stack(encoder_outputs, dim=1)  # [B, K, L, D]
        
        # Squeeze operation
        Z = self.squeeze(U).squeeze(-1)  # [B, K, L]
        Z = Z.mean(dim=2)  # [B, K]
        
        # Excitation operation
        s = self.W2(F.relu(self.W1(Z)))  # [B, K]
        s = torch.sigmoid(s)  # [B, K]
        
        # Apply attention weights
        U_weighted = U * s.unsqueeze(-1).unsqueeze(-1)  # [B, K, L, D]
        
        # Sum across layers
        U_sum = U_weighted.sum(dim=1)  # [B, L, D]
        
        # Final projection
        output = self.output_projection(U_sum)  # [B, L, output_size]
        
        return output