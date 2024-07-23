from typing import List, Dict, Any
from PIL import Image
import random
import torch

class DataCollator:
    def __init__(self, tokenizer, feature_extractor, max_objects=10, 
                    neg_rate=1, max_length = 16, resize_image=False,
                                                    num_query_groups=1,
                                                    add_no_object=False):
        self.tokenizer=tokenizer
        self.feature_extractor=feature_extractor
        self.max_objects=max_objects
        self.neg_rate = neg_rate
        self.max_length = max_length
        self.resize_image = resize_image
        self.num_query_groups = num_query_groups
        self.add_no_object=add_no_object

    def get_all_nouns(self, batch):
        all_nouns = set()
        for item in batch:
            id2chunk = item['id2chunk']
            nouns = set(id2chunk.values())
            all_nouns.update(nouns)
        all_nouns = list(all_nouns)
        if self.add_no_object:
            all_nouns.append('no object')
        return all_nouns
    
    def tokenize(self, texts, image):
        model_inputs = self.tokenizer(texts, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Get the target size from the feature extractor's configuration
        if self.resize_image:
            image = image.resize((256, 256), Image.BILINEAR)
        try:
            image_features = self.feature_extractor(images=image, return_tensors="pt")
        except:
            return None
        pixel_values = image_features['pixel_values'][0]
        image_features['pixel_values'] = pixel_values
        model_inputs.update(image_features)
        return model_inputs
    
    def get_labels(self, item, chunks):
        chunk2id = {chunk:id for id, chunk in enumerate(chunks)}
        bboxs = []
        labels = []
        for id, bbox in item['id2bbox'].items():
            chunk = item['id2chunk'][id]
            label = chunk2id[chunk]
            bboxs.append(bbox)
            labels.append(label)
        return {"boxes": torch.tensor(bboxs), 'labels': torch.tensor(labels)}

    def process_item(self, item, all_chunks):
        model_inputs = self.tokenize(all_chunks, item['image'])    
        if model_inputs is None:
            return None
        labels = self.get_labels(item, all_chunks)
        model_inputs.update(labels)
        return model_inputs

    def porcess_batch(self, batch):
        batch = [item for item in batch if item is not None]
        all_chunks = self.get_all_nouns(batch)
        features = []
        for item in batch:
            try:
                model_inputs = self.process_item(item, all_chunks)
            except:
                continue
            if model_inputs:
                features.append(model_inputs)
        return features
    
    def __call__(self, batch: List[Dict[str, Any]]):
        features = self.porcess_batch(batch)
        if len(features)==0:
            return None
        first = features[0]
        batch = {}

        targets = [{'labels': feature['labels'], 'boxes': feature['boxes']} for feature in features]
        for k in first:
            if k in {'labels', 'boxes', 'pixel_mask'}:
                continue
            x = torch.stack([f[k] for f in features])
            if k in {'input_ids', 'attention_mask'}:
                x = x.view(-1, x.shape[-1])
            elif k == 'pixel_values':
                x = x.squeeze(1)
            batch[k] = x
        targets*=self.num_query_groups
        batch['targets'] = targets
        return batch
