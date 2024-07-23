from typing import Optional, List
from pathlib import Path
import torch

from .config import GliDerConfig

class ModelWraper(torch.nn.Module):
    def __init__(self, model, matcher=None, criterion=None, processor = None):
        super().__init__()
        self.backbone = model
        self.matcher = matcher
        self.criterion = criterion
        self.processor = processor

    @property
    def device(self):
        return self.backbone.device
    
    def forward(self, input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[List] = None,
        **kwargs):

        outputs = self.backbone(input_ids=input_ids,
                             pixel_values=pixel_values,
                             attention_mask=attention_mask,
                             return_dict=True,
                             **kwargs)
        if targets is not None:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)# loss_dict['loss_ce']s
            return (losses, outputs)
        else:
            return outputs

    def resize_token_embeddings(self, add_tokens, 
                                    set_class_token_index = True, 
                                    add_tokens_to_tokenizer = True, 
                                    pad_to_multiple_of=None) -> torch.nn.Embedding:
        if set_class_token_index:
            self.backbone.config.class_token_index = len(self.processor.tokenizer)
        if add_tokens_to_tokenizer:
            self.processor.tokenizer.add_tokens(add_tokens)
        new_num_tokens = len(self.processor.tokenizer)
        model_embeds = self.backbone.text_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.backbone.config.text_config.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def save_pretrained(
            self,
            save_directory: str,
            *,
            config: Optional[GliDerConfig] = None,
            repo_id: Optional[str] = None,
            push_to_hub: bool = False,
            **push_to_hub_kwargs,
    ) -> Optional[str]:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        torch.save(self.backbone.state_dict(), save_directory / "pytorch_model.bin")
        # save config (if provided)
        if config is None:
            config = self.backbone.config
        if config is not None:
            config.to_json_file(save_directory / "config.json")
        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None