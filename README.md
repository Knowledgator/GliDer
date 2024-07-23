# GliDer: Generalist Language-Image Detector
GliDer is an advanced zero-shot object detection framework that combines state-of-the-art vision and language models to detect objects in images without prior training on specific object classes.

## Key Features

- Support for various image backbones
- Support for different language backbones
- OWL (Object-Wise Learning) training support
- Image encoders and decoders support
- Deep fusion of image and text information
- Vision layers fusion
- Layers groups for better vision encoder training
- Support for different post-fusion schemas
- Objectness loss support
- Bounding box (bbox) loss support
- Generalized Intersection over Union (GIOU) loss support
- Cross-entropy and Varifocal Loss (VFL) support for classification

## Requirements

(List the required libraries and their versions here. You can generate this list using `pip freeze > requirements.txt` and then include the key dependencies.)

## How to Train

To train the GliDer model, use the `train.py` script. Here's an example command:

```bash
python train.py --save_path models/highres --data_path data/grit_high_res --batch_size 64 --num_epochs 12 --image_size 768 --lr 3e-5
```

For a full list of training options, run:

```bash
python train.py --help
```

## Format of the Dataset

GliDer uses the WebDataset format for efficient data loading. Your dataset should be structured as follows:

1. Create tar files containing 1000 images each, along with their JSON annotation files.
2. Name the tar files numerically (e.g., 00000.tar, 00001.tar, etc.).
3. The JSON annotation file for each image should have the following structure:

```json
{
  "caption": "A dog playing with a frisbee in the park",
  "noun_chunks": [
    [0, 5, 0.1, 0.2, 0.5, 0.6],  # "A dog" with bounding box [ 0.1, 0.2, 0.5, 0.6]
    [18, 25, 0.3, 0.4, 0.7, 0.8]  # "frisbee" with bounding box [0.3, 0.4, 0.7, 0.8]
  ]
}
```

4. Organize your data directory as follows:

```
data/
├── dataset_name/
│   ├── 00000.tar
│   ├── 00001.tar
│   └── ...
```

### GRIT Dataset
In our study we used [GRIT](https://huggingface.co/datasets/zzliang/GRIT) dataset that has required structure

To download the dataset follow the instruction below:

1. Download the metadata. You can download it by cloning current repository:
```bash
git lfs install
git clone https://huggingface.co/datasets/zzliang/GRIT
```
2. Install [img2dataset](https://github.com/rom1504/img2dataset).
```bash
pip install img2dataset
```
3. Download images
You need to replace `/path/to/GRIT_dataset/grit-20m` with the local path to this repository. 
```bash
img2dataset --url_list /path/to/GRIT_dataset/grit-20m --input_format "parquet"\
    --url_col "url" --caption_col "caption" --output_format webdataset \
    --output_folder /tmp/grit --processes_count 4 --thread_count 64 --image_size 256 \
    --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
    --save_additional_columns '["id","noun_chunks","ref_exps","clip_similarity_vitb32","clip_similarity_vitl14"]' \
    --enable_wandb False
```
You can adjust some parameters according to your actual needs (e.g., `processes_count`, `thread_count`, `image_size`, `save_additional_columns`).
More img2dataset hyper-parameters can be found in [here](https://github.com/rom1504/img2dataset#api).


## How to Test Model

To test the trained model, use the `test.py` script. Here's an example of how to use it:

```python
import argparse
from glider import GliDer
from glider.config import GliDerConfig
from glider.processing import GliDerProcessor
from PIL import Image
import requests
import torch

# Set up argument parser and model configuration
parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', type=str, default='models/base/best')
args = parser.parse_args()

config = GliDerConfig.from_pretrained(args.model_checkpoint)
model = GliDerForObjectDetection.from_pretrained(args.model_checkpoint, config=config).to('cuda:0')
model.eval()

tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name, size=config.vision_config.image_size)
processor = GliDerProcessor(tokenizer=tokenizer, image_processor=image_processor)

# Perform object detection
def detect(text, url, threshold=0.5):
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=text, images=image, truncation=True, max_length=10).to('cuda:0')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_object_detection_overlap(outputs=outputs, threshold=threshold, target_sizes=torch.Tensor([image.size[::-1]]))
    
    # Process and display results
    for box, score, label, obj in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"], results[0]["objectness"]):
        if (score * obj) > threshold and score > 0.5:
            print(f"Detected {text[label]} with confidence {score:.3f} and objectness {obj:.3f} at location {box.tolist()}")

# Example usage
text_queries = ["a dog", "a cat", "a frisbee"]
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
detect(text_queries, image_url, threshold=0.5)
```

This script demonstrates how to load a trained model, process an image, and perform object detection with text queries.

## Contributing

We welcome contributions to the GliDer project! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with clear, descriptive messages
4. Push your changes to your fork
5. Create a pull request to the main repository

Please ensure that your code adheres to the project's coding standards and includes appropriate tests and documentation.

For major changes or new features, please open an issue first to discuss the proposed changes.

Thank you for helping improve GliDer!