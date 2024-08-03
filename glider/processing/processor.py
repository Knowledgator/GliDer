from typing import List, Union, Tuple

import torch
import torch.nn.functional as F

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def box_iou(box1, box2):
    """
    Compute the IoU of two sets of boxes.
    Args:
        box1: (Tensor[N, 4])
        box2: (Tensor[M, 4])
    Returns:
        iou: (Tensor[N, M]): the NxM matrix containing the pairwise IoU values
    """
    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def box_area(boxes):
    """Compute the area of a set of bounding boxes"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression"""
    boxes = box_cxcywh_to_xyxy(boxes)

    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.numel() == 1:
            break
        iou = box_iou(boxes[i:i+1], boxes[idxs[1:]])
        idxs = idxs[1:][iou[0] <= iou_threshold]
    return torch.LongTensor(keep)

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - w / 2), (cy - h / 2),
         (cx + w / 2), (cy + h / 2)]
    return torch.stack(b, dim=-1)


def post_process_object_detection(
    outputs, threshold: float = 0.1, target_sizes: Union[torch.TensorType, List[Tuple]] = None, 
                                            use_nms = True, iou_threshold = 0.3, multi_label=True
):
    """
    Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
    bottom_right_x, bottom_right_y) format.

    Args:
        outputs ([`OwlViTObjectDetectionOutput`]):
            Raw outputs of the model.
        threshold (`float`, *optional*):
            Score threshold to keep object detection predictions.
        target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
            Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
            `(height, width)` of each image in the batch. If unset, predictions will not be resized.
    Returns:
        `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
        in the batch as predicted by the model.
    """

    logits, boxes, objectness = outputs.logits, outputs.pred_boxes, outputs.objectness_logits
    
    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )
    if multi_label:
        probs = torch.sigmoid(logits)
        best = torch.max(probs, dim=-1)
        scores = best.values
        labels = best.indices
    else:
        best = torch.max(torch.softmax(logits, dim=-1), dim=-1)
        scores = best.values
        labels = best.indices

    objectness = torch.sigmoid(objectness)

    if target_sizes is not None:
        if isinstance(target_sizes, List):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b, o in zip(scores, labels, boxes, objectness):
        # Filter based on score threshold
        mask = s >= threshold
        score = s[mask]
        label = l[mask]
        box = b[mask]
        obj = o[mask]
        
        if use_nms:
            keep = nms(box, score, iou_threshold)
            
            score = score[keep]
            label = label[keep]
            box = box[keep]
            obj = obj[keep]

        results.append({
            "scores": score,
            "labels": label,
            "boxes": box,
            'objectness': obj
        })

    return results

def post_process_object_detection_grid_relative(
    outputs, threshold: float = 0.1, target_sizes: Union[torch.TensorType, List[Tuple]] = None,
    iou_threshold: float = 0.3, grid_size=(14, 14)
):

    logits, boxes, objectness = outputs.logits, outputs.pred_boxes, outputs.objectness_logits
    
    if target_sizes is not None and len(logits) != len(target_sizes):
        raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
    
    best = torch.max(torch.softmax(logits, dim=-1), dim=-1)
    scores = best.values
    labels = best.indices
    
    queries = torch.arange(logits.shape[1]).to(logits.device)
    xg = (queries%grid_size[0]).unsqueeze(0)
    yg = (queries//grid_size[1]).unsqueeze(0)
    
    boxes[:, :, 0] /= grid_size[0]
    boxes[:, :, 1] /= grid_size[1]
    boxes[:, :, 0] += xg/grid_size[0]
    boxes[:, :, 1] += yg/grid_size[1]

    objectness = torch.sigmoid(objectness)
    
    # Convert from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        if isinstance(target_sizes, List):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b, o in zip(scores, labels, boxes, objectness):
        # Filter based on score threshold
        mask = s >= threshold
        score = s[mask]
        label = l[mask]
        box = b[mask]
        obj = o[mask]
                # Apply NMS
        keep = nms(box, score, iou_threshold)

        results.append({
            "scores": score[keep],
            "labels": label[keep],
            "boxes": box[keep],
            'objectness': obj[keep]
        })
    
    return results

class GliDerProcessor:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, padding="max_length", **kwargs):
        """
        Main method to prepare for the model one or several text(s) and image(s). This method forwards the `text` and
        `kwargs` arguments to CLIPTokenizerFast's [`~CLIPTokenizerFast.__call__`] if `text` is not `None` to encode:
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
            `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if text is None or images is None:
            raise ValueError(
                "You have to specify text and image."
            )

        if text is not None:
            if isinstance(text, str) or (isinstance(text, List) and not isinstance(text[0], List)):
                encodings = [self.tokenizer(text, padding=padding, return_tensors='pt', **kwargs)]

            elif isinstance(text, List) and isinstance(text[0], List):
                encodings = []

                # Maximum number of queries across batch
                max_num_queries = max([len(t) for t in text])

                # Pad all batch samples to max number of text queries
                for t in text:
                    if len(t) != max_num_queries:
                        t = t + [" "] * (max_num_queries - len(t))

                    encoding = self.tokenizer(t, padding=padding, return_tensors='pt', **kwargs)
                    encodings.append(encoding)
            else:
                raise TypeError("Input text should be a string, a list of strings or a nested list of strings")

            input_ids = torch.cat([encoding["input_ids"] for encoding in encodings], dim=0)
            attention_mask = torch.cat([encoding["attention_mask"] for encoding in encodings], dim=0)

            encoding = BatchEncoding()
            encoding["input_ids"] = input_ids
            encoding["attention_mask"] = attention_mask

        if images is not None:
            image_features = self.image_processor(images, return_tensors='pt', **kwargs)

        encoding["pixel_values"] = image_features.pixel_values
        return encoding
    
    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_object_detection with OWLViT->OWLv2
    def post_process_object_detection(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`post_process_object_detection`]. Please refer
        to the docstring of this function for more information.
        """
        return post_process_object_detection(*args, **kwargs)

    def post_process_object_detection_grid_relative(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`post_process_object_detection`]. Please refer
        to the docstring of this function for more information.
        """
        return post_process_object_detection_grid_relative(*args, **kwargs)
    
    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.batch_decode
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.decode
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)