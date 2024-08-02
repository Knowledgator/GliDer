import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from glider.modeling import GliDer
from glider.config import GliDerConfig
from glider.processing import GliDerProcessor
from transformers import AutoTokenizer, AutoImageProcessor

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b

def detect(image_path, text, processor, model, threshold=0.5):
    image = Image.open(image_path)
    inputs = processor(text=text, images=image, truncation=True, max_length=10).to('cuda:0')
    
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    
    return boxes, scores, labels

def coco_evaluate(args):
    # Load COCO dataset
    coco_gt = COCO(args.ann_file)
    image_ids = coco_gt.getImgIds()

    # Load model and processor
    config = GliDerConfig.from_pretrained(args.model_checkpoint)
    model = GliDer.from_pretrained(args.model_checkpoint, ignore_mismatched_sizes=True, config=config).to('cuda:0')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    tokenizer.model_input_names = ['input_ids', 'attention_mask']

    image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name, size=config.vision_config.image_size)
    processor = GliDerProcessor(tokenizer=tokenizer, image_processor=image_processor)

    # Prepare category names for detection
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_names = [cat['name'] for cat in categories]

    # Perform detection on all images
    results = []
    for image_id in tqdm(image_ids):
        img_info = coco_gt.loadImgs(image_id)[0]
        image_path = Path(args.image_dir) / img_info['file_name']
        
        boxes, scores, labels = detect(image_path, category_names, processor, model, args.threshold)
        
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box_cxcywh_to_xyxy(box.tolist())
            result = {
                'image_id': image_id,
                'category_id': coco_gt.getCatIds(catNms=[category_names[label]])[0],
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'score': score.item()
            }
            results.append(result)

    # Save results
    with open(args.results_file, 'w') as f:
        json.dump(results, f)

    # Evaluate using COCO API
    coco_dt = coco_gt.loadRes(args.results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='models/base')
    parser.add_argument('--ann_file', type=str,  help='Path to COCO annotation file', default = 'data/coco/annotations')
    parser.add_argument('--image_dir', type=str, help='Path to COCO image directory',  default = 'data/coco/val2017')
    parser.add_argument('--results_file', type=str, default='coco_results.json', help='Path to save detection results')
    parser.add_argument('--threshold', type=float, default=0.1, help='Detection threshold')
    args = parser.parse_args()

    coco_evaluate(args)