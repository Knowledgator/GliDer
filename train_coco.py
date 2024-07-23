import os
import torch
import argparse
import torchvision
from typing import List, Any, Dict, Optional
from transformers import AutoFeatureExtractor
from transformers.utils import logging
from glider.modeling import SupervisedGliDer
from glider.config import GliDerConfig
from glider.loss import HungarianMatcher, SetCriterion
from glider.training import Trainer, TrainingArguments

logger = logging.get_logger(__name__)

class ModelWraper(torch.nn.Module):
    def __init__(self, model, matcher, criterion):
        super().__init__()
        self.backbone = model
        self.matcher = matcher
        self.criterion = criterion

    @property
    def device(self):
        return self.backbone.device
    
    def forward(self,
        pixel_values: torch.FloatTensor,
        targets: Optional[List] = None):

        outputs = self.backbone(
                             pixel_values=pixel_values,
                             return_dict=True)
        if targets is not None:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)# loss_dict['loss_ce']s
            return (losses, outputs)
        else:
            return outputs
    
    def save_pretrained(self, output_dir, **kwargs):
        """
        Save the model and its configuration to the directory `output_dir`.
        """
        self.backbone.save_pretrained(output_dir, **kwargs)


class DataCollator:
    def __init__(self, max_objects=10):
        self.max_objects=max_objects

    def __call__(self, features: List[Dict[str, Any]]):
        first = features[0]
        batch = {}

        targets = [{'labels': feature['labels'], 'boxes': feature['boxes']} for feature in features]
        for k in first:
            if k in {'labels', 'boxes'}:
                continue
            x = torch.stack([f[k] for f in features])
            x = x.squeeze(1)
            batch[k] = x
        batch['targets'] = targets
        return batch

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, feature_extractor):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor
    
    def process_bbox(self, bbox, image):
        x1, y1, w, h = bbox
        x2 = x1+w/2
        y2 = y1+h/2
        width, height = image.size
        return [x2/width, y2/height, w/width, h/height]

    def __getitem__(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        w, h = image.size
        boxes = []
        labels = []
        for object in target[:args.max_objects]:
            if 'iscrowd' not in object or object['iscrowd'] == 0:
                bbox = self.process_bbox(object['bbox'], image)
                boxes.append(bbox)
                label = object['category_id']-1
                labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes.clamp_(min=0, max=1)

        labels = torch.tensor(labels, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]

        image_features = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = image_features['pixel_values'][0]
        image_features['pixel_values'] = pixel_values

        image_features['boxes'] = boxes
        image_features['labels'] = labels
        return image_features
    

def main(args):
    if args.from_pretrained:
        configs = GliDerConfig.from_pretrained(args.from_pretrained)
        feature_extractor = AutoFeatureExtractor.from_pretrained(configs.vision_model_name)
        model = SupervisedGliDer.from_pretrained(args.from_pretrained)
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.vision_model)
        configs = GliDerConfig(vision_model_name = args.vision_model)
        model = SupervisedGliDer(configs, num_labels = args.num_labels+1, from_pretrained=True)

    
    if args.freeze_vision_model:
        model.vision_model.requires_grad_(False)

    train_dataset = CocoDetection(args.train_images_data_path, args.train_annotations_data_path, feature_extractor)
    test_dataset = CocoDetection(args.val_images_data_path, args.val_annotations_data_path, feature_extractor)

    data_collator = DataCollator()

    matcher = HungarianMatcher(cost_class=args.matcher_cost_class, 
                                    cost_bbox=args.matcher_cost_box, 
                                            cost_giou=args.matcher_cost_giou)

    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef, 
                                                    'loss_giou':args.giou_loss_coef}

    criterion = SetCriterion(num_classes=args.num_labels, matcher=matcher, 
                                        eos_coef=args.eos_coef, weight_dict=weight_dict,
                                        focal_loss = args.focal_loss)

    model = ModelWraper(model, matcher, criterion)

    training_args = TrainingArguments(
        output_dir=args.save_path,
        learning_rate=args.lr,
        vision_model_lr=args.vision_model_lr,
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
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="wandb" if args.wandb else "none"

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=feature_extractor,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='models/coco')
    parser.add_argument('--train_annotations_data_path', type=str, default='data/coco/annotations/instances_train2017.json')
    parser.add_argument('--val_annotations_data_path', type=str, default='data/largedir/coco/annotations/instances_val2017.json')    
    parser.add_argument('--train_images_data_path', type=str, default='data/largedir/coco/train2017')
    parser.add_argument('--val_images_data_path', type=str, default='data/largedir/coco/val2017')        
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_labels', type=int, default=90)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vision_model', type=str, default='openai/clip-vit-base-patch16')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--max_saves', type=int, default=3)
    parser.add_argument('--save_iters', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--max_objects', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vision_model_lr', type=float, default=1e-5)
    parser.add_argument('--num_negatives', type=int, default=1)
    parser.add_argument('--vision_model_weight_decay', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--freeze_vision_model', type=bool, default=False)
    parser.add_argument('--deep_fusion', type=bool, default=False)
    parser.add_argument('--eos_coef', type=float, default=0.1)
    parser.add_argument('--focal_loss', type=bool, default=False)
    parser.add_argument('--focal_alpha', type=float, default=0.3)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--bbox_loss_coef', type=float, default=1.0)
    parser.add_argument('--giou_loss_coef', type=float, default=1.0)
    parser.add_argument('--ce_loss_coef', type=float, default=1.0)
    parser.add_argument('--matcher_cost_class', type=float, default=1.0)
    parser.add_argument('--matcher_cost_box', type=float, default=5.0)
    parser.add_argument('--matcher_cost_giou', type=float, default=2.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--dataloader_num_workers', type=int, default=12)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--from_pretrained', type=str, default='')
    args = parser.parse_args()

    main(args)