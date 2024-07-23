from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.clip.configuration_clip import CLIPTextConfig
from owl.config import Owlv2VisionConfig, Owlv2TextConfig

class GliDerConfig(PretrainedConfig):
    model_type = "glider"
    is_composition = True
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        vision_model_name=None,
        text_model_name=None,
        deep_fusion=None,
        fusion_strategy='full',#mean
        post_fusion_schema = '', #l2l-l2i
        text_layers_fustion=False,
        vision_layers_fusion=False,
        is_vit_det=False,
        decoder_layers=0,
        num_queries=300,
        ignore_index=-100,
        projector_hidden_act="gelu",
        zero_class_embedding=True,
        initializer_factor = 0.03,
        class_token_index = -1,
        max_objects = 25,
        fusion_dropout = 0.3,
        num_query_groups=3,
        has_box_bias=True, 
        image_size=None,
        patch_size=None,
        **kwargs,
    ):
        self.image_size=image_size
        self.patch_size=patch_size
        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.decoder_layers=decoder_layers
        self.is_vit_det=is_vit_det
        self.num_queries=num_queries
        self.has_box_bias=has_box_bias
        self.deep_fusion = deep_fusion
        self.fusion_strategy = fusion_strategy
        self.text_layers_fustion = text_layers_fustion
        self.vision_layers_fusion = vision_layers_fusion
        self.zero_class_embedding = zero_class_embedding
        self.initializer_factor = initializer_factor
        self.class_token_index = class_token_index
        self.max_objects = max_objects
        self.fusion_dropout = fusion_dropout
        self.post_fusion_schema = post_fusion_schema
        self.num_query_groups = num_query_groups
        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            if vision_config['model_type'] == 'owlv2_vision_model':
                vision_config = Owlv2VisionConfig(**vision_config)
            else:
                vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "bert"
            if text_config["model_type"]=='clip_text_model':
                text_config = CLIPTextConfig(**text_config)
            elif text_config['model_type'] == 'owlv2_text_model':
                text_config = Owlv2TextConfig(**text_config)
            else:
                text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        self.text_config = text_config

        super().__init__(**kwargs)
        
    def __setattr__(self, key, value):
        try:
            if key in super().__getattribute__("attribute_map"):
                key = super().__getattribute__("attribute_map")[key]
            super().__setattr__(key, value)
        except:
            pass

    @property
    def vocab_size(self):
        return self.text_config.vocab_size

    @property
    def text_hidden_size(self):
        return self.text_config.hidden_size
    
    @property
    def vision_hidden_size(self):
        return self.vision_config.hidden_size
    
    @property
    def image_size(self):
        return self.vision_config.image_size

    @property
    def patch_size(self):
        return self.vision_config.patch_size

    def to_dict(self):
        output = super().to_dict()
        return output