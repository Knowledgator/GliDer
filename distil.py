import os
import io
import json
import argparse
import webdataset as wds
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, Owlv2Processor
from glider.owl.model import Owlv2ForObjectDetection
from glider.owl.config import Owlv2Config

from glider.modeling import GliDer
from glider.config import GliDerConfig
from glider.loss import DistilationCriterion
from glider.training import Trainer, TrainingArguments
from glider.processing.data_collator import DistilationDataCollator
from glider import DistillationModelWrapper


def main(args):
    if args.teacher_type == 'owl':
        teacher_model = Owlv2ForObjectDetection.from_pretrained(args.teacher_model)
        teacher_tokenizer = Owlv2Processor.from_pretrained(args.teacher_model)
        processor = Owlv2Processor.from_pretrained(args.teacher_model)
        teacher_feature_extractor = processor.image_processor
    else:
        configs = GliDerConfig.from_pretrained(args.teacher_model)

        teacher_model = GliDer.from_pretrained(args.teacher_model)

        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        teacher_feature_extractor = AutoImageProcessor.from_pretrained(configs.vision_model_name, size=args.image_size)


    if args.model_type == 'owl':
        configs = Owlv2Config.from_pretrained(args.from_pretrained, 
                                            post_fusion_schema = args.post_fusion_schema)
        
        if hasattr(configs.vision_config, 'image_size') and args.image_size !=None:
            configs.vision_config.image_size = args.image_size
        
        if hasattr(configs.vision_config, 'patch_size') and args.patch_size!=None:
            configs.vision_config.patch_size = args.patch_size

        model = Owlv2ForObjectDetection.from_pretrained(args.from_pretrained, config=configs)
        tokenizer = Owlv2Processor.from_pretrained(args.from_pretrained)
        processor = Owlv2Processor.from_pretrained(args.from_pretrained)
        feature_extractor = processor.image_processor
    else:
        if args.from_pretrained:
            tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
            configs = GliDerConfig.from_pretrained(args.from_pretrained)

            if hasattr(configs.vision_config, 'image_size') and args.image_size !=None:
                configs.vision_config.image_size = args.image_size
            
            if hasattr(configs.vision_config, 'patch_size') and args.patch_size!=None:
                configs.vision_config.patch_size = args.patch_size

            if 'owl' in configs.vision_model_name:
                processor = Owlv2Processor.from_pretrained(configs.vision_model_name)
                feature_extractor = processor.image_processor
            else:
                feature_extractor = AutoImageProcessor.from_pretrained(configs.vision_model_name, size=args.image_size)
            model = GliDer.from_pretrained(args.from_pretrained, config = configs, ignore_mismatched_sizes=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.language_model)
            if 'owl' in args.vision_model:
                processor = Owlv2Processor.from_pretrained(args.vision_model)
                feature_extractor = processor.image_processor
            else:
                feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model, size=args.image_size) #AutoFeatureExtractor
            configs = GliDerConfig(text_model_name = args.language_model, vision_model_name = args.vision_model,
                                                                                    deep_fusion=args.deep_fusion,
                                                                                    vision_layers_fusion=args.vision_layers_fusion,
                                                                                    num_query_groups=args.num_query_groups,
                                                                                    post_fusion_schema=args.post_fusion_schema,
                                                                                    is_vit_det=args.is_vit_det,
                                                                                    has_box_bias=args.has_box_bias)
            model = GliDer(configs, from_pretrained=True, image_size=args.image_size, patch_size=args.patch_size)
        tokenizer.model_input_names = ['input_ids', 'attention_mask']

    if args.freeze_language_model:
        model.text_model.requires_grad_(False)
    if args.freeze_vision_model:
        model.vision_model.requires_grad_(False)
    
    def box_xyxy_to_cxcywh(box):
        x0, y0, x1, y1 = box
        box = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return box
    
    def get_noun_chunks(metadata):
        id2bbox = {}
        id2chunk = {}
        caption = metadata['caption']
        for id, chunk in enumerate(metadata['noun_chunks'][:args.max_objects]):
            start, end = chunk[:2]
            span = caption[int(start):int(end)]
            id2chunk[id] = span
            box = chunk[2:6]
            if args.box_format == 'cxcywh':
                box = box_xyxy_to_cxcywh(box)
            id2bbox[id] = box
        return id2chunk, id2bbox

    def custom_decode(sample):
        """Decode an entire sample.

        :param sample: the sample, a dictionary of key value pairs
        """
        result = {}
        assert isinstance(sample, dict), sample
        for k, v in list(sample.items()):
            if isinstance(v, bytes) and k == 'jpg':
                try:
                    img = Image.open(io.BytesIO(v))
                    img.verify()  # Verify image validity
                    img = Image.open(io.BytesIO(v))  # Re-open the image to use it later
                    img = img.convert("RGB")
                except (IOError, OSError, UnidentifiedImageError) as e:
                    print(f"Error decoding image: {e}")
                    img = None
                result[k] = img
            elif isinstance(v, bytes) and k == 'json':
                metadata = json.loads(v.decode("utf-8"))
                result[k] = metadata
            else:
                result[k] = v
        return result

    def preprocess(example):
        image = example['jpg']
        metadata = example['json']
        if metadata is None or image is None:
            return None
        try:
            # Attempt to verify the image again to ensure it can be processed
            if isinstance(image, Image.Image):
                image.verify()
            else:
                return None
            id2chunk, id2bbox = get_noun_chunks(metadata)
            return {"image": image, "id2chunk": id2chunk, "id2bbox": id2bbox}
        except (IOError, OSError, UnidentifiedImageError):
            # If there's an error with the image, return None
            return None


    train_dataset = (wds.WebDataset(args.data_path+'/{00001..01999}.tar')
                                            .map(custom_decode)
                                            .map(preprocess))
    
    test_dataset = (wds.WebDataset(args.data_path+'/00000.tar')
                                            .map(custom_decode)
                                            .map(preprocess))

    data_collator = DistilationDataCollator(teacher_tokenizer, teacher_feature_extractor, 
                                    tokenizer, feature_extractor, resize_image=True, 
                                        num_query_groups=args.num_query_groups,
                                        add_no_object = args.add_no_object)

    weight_dict = {
                   'classification': args.classification_loss_coef, 
                   'objectness': args.objectness_loss_coef,
                   'boxes': args.boxes_loss_coef,
                   'giou': args.giou_loss_coef
    }

    criterion = DistilationCriterion(temperature=args.temperature, 
                                        weight_dict=weight_dict)

    model = DistillationModelWrapper(model, teacher_model, criterion)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        learning_rate=args.lr,
        language_model_lr=args.language_model_lr,
        vision_model_lr=args.vision_model_lr,
        language_model_weight_decay=args.language_model_weight_decay,
        vision_model_weight_decay=args.vision_model_weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_steps = args.save_iters,
        save_total_limit=args.max_saves,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="wandb" if args.wandb else "none",
        fp16=True, 
        max_grad_norm=1.,
        use_cpu = False,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='models/distilled')
    parser.add_argument('--data_path', type=str, default='data/grit_high_res')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--teacher_type', type=str, default='owl')
    parser.add_argument('--teacher_model', type=str, default='google/owlv2-base-patch16')
    parser.add_argument('--model_type', type=str, default='')#owl
    parser.add_argument('--vision_model', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('--language_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--is_vit_det', type=bool, default=False)
    parser.add_argument('--has_box_bias', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=12)
    parser.add_argument('--max_saves', type=int, default=5)
    parser.add_argument('--save_iters', type=int, default=3000)
    parser.add_argument('--max_steps', type=int, default=500000)
    parser.add_argument('--max_objects', type=int, default=25)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=None)
    parser.add_argument('--temperature', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--language_model_lr', type=float, default=1e-5)
    parser.add_argument('--vision_model_lr', type=float, default=1e-5)
    parser.add_argument('--num_negatives', type=int, default=10)
    parser.add_argument('--language_model_weight_decay', type=float, default=0.1)
    parser.add_argument('--vision_model_weight_decay', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--freeze_language_model', type=bool, default=False)
    parser.add_argument('--freeze_vision_model', type=bool, default=False)
    parser.add_argument('--deep_fusion', type=bool, default=False)
    parser.add_argument('--vision_layers_fusion', type=bool, default=False)
    parser.add_argument('--num_query_groups', type=int, default=1)
    parser.add_argument('--post_fusion_schema', type=str, default='') #l2i-i2l-l2l
    parser.add_argument('--add_no_object', type=bool, default=True)
    parser.add_argument('--box_format', type=str, default="cxcywh") #cxcywh
    parser.add_argument('--classification_loss_coef', type=float, default=1.0)
    parser.add_argument('--objectness_loss_coef', type=float, default=1.0)
    parser.add_argument('--boxes_loss_coef', type=float, default=1.0)
    parser.add_argument('--giou_loss_coef', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--dataloader_num_workers', type=int, default=12)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--from_pretrained', type=str, default='')
    args = parser.parse_args()

    main(args)