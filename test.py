import argparse
from glider.modeling import GliDer
from glider.config import GliDerConfig
from glider.processing import GliDerProcessor
from PIL import Image, ImageDraw
import requests
import torch

from transformers import AutoTokenizer, AutoImageProcessor

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b

def detect(text, url, threshold=0.5):
    print('Testing processing...')

    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=text, images=image, truncation=True, max_length=10).to('cuda:0')
    
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])

    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    objectness = results[i]["objectness"]
    # Print detected objects and rescaled box coordinates
    draw = ImageDraw.Draw(image)
    for box, score, label, obj in zip(boxes, scores, labels, objectness):
        label = label.item()
        obj = obj.item()
        score = score.item()
        if (score*obj)>threshold and score>0.5:
            print('Yes', score*obj, score, obj, text[label])
            tag = text[label]
            if tag =='no object':
                continue
            box = [round(i, 2) for i in box.tolist()]
            box = box_cxcywh_to_xyxy(box)
            print(f"Detected {tag} with confidence {round(score, 3)} and objecntess {round(obj, 3)} at location {box}")
            try:
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]), tag, fill="red", font_size=32)
            except:
                continue
    image.save("detected_objects.jpg")


def start_detection_loop():
    prev_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    prev_texts = ["a dog", "a cat", "a vehicle", 'a building', "a bird", 'remote', 'no object', 'a person']
    prev_threshold = 0.5

    while True:
        print('Treshhold: ')
        threshold = input()
        if not threshold:
            threshold = prev_threshold
        else:
            threshold = float(threshold)
        print('Texts:')
        texts = input()
        if not texts:
            texts = prev_texts
        else:
            texts = texts.split(', ')
        print('URL: ')
        url = input()
        if not url:
            url = prev_url
        detect(texts, url, threshold)

        prev_texts = texts
        prev_url = url
        prev_threshold = threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='models/base/best')
    parser.add_argument('--save_path', type=str, default='test_images')
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=None)
    args = parser.parse_args()

    config = GliDerConfig.from_pretrained(args.model_checkpoint)

    if args.image_size is None:
        image_size = config.vision_config.image_size
    else:
        image_size = args.image_size

    if args.patch_size is None:
        patch_size = config.vision_config.patch_size
    else:
        patch_size = args.patch_size

    if hasattr(config.vision_config, 'image_size') and image_size!=None:
        config.vision_config.image_size = image_size
    
    if hasattr(config.vision_config, 'patch_size') and patch_size!=None:
        config.vision_config.patch_size = patch_size

    model = GliDer.from_pretrained(args.model_checkpoint,
                                                        ignore_mismatched_sizes=True,
                                                        config=config).to('cuda:0')

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    tokenizer.model_input_names = ['input_ids', 'attention_mask']

    image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name, size=image_size)

    processor = GliDerProcessor(tokenizer=tokenizer, image_processor=image_processor)

    start_detection_loop()
