import json
import numpy as np
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def compute_ap50(ground_truths, predictions, iou_threshold=0.7):
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)

    for gt in ground_truths['annotations']:
        gt_by_image[gt['image_id']].append(gt)

    for pred in predictions:
        pred_by_image[pred['image_id']].append(pred)

    all_gt_boxes = []
    all_pred_boxes = []

    for image_id in gt_by_image:
        gt_boxes = [gt['bbox'] for gt in gt_by_image[image_id]]
        pred_boxes = [pred['bbox'] for pred in pred_by_image.get(image_id, [])]
        gt_detected = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                if compute_iou(gt_box, pred_box) >= iou_threshold:
                    if not gt_detected[i] and not pred_matched[j]:
                        gt_detected[i] = True
                        pred_matched[j] = True
                        break

        all_gt_boxes.extend(gt_detected)
        all_pred_boxes.extend(pred_matched)

    tp = sum(all_pred_boxes)
    fp = len(all_pred_boxes) - tp
    fn = len(all_gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

def average_precision(ground_truths, predictions, conf_thresholds):
    precision_list = []
    recall_list = []
    for conf_threshold in conf_thresholds:
        filtered_predictions = [pred for pred in predictions if pred['score'] >= conf_threshold]
        precision, recall = compute_ap50(ground_truths, filtered_predictions)
        precision_list.append(precision)
        recall_list.append(recall)

    return np.mean(precision_list), np.mean(recall_list)

# Load the JSON files
ground_truths = load_json('/mnt/workspace/MusicYOLO/datasets/images/valid/_annotations.coco.json')
predictions = load_json('/mnt/workspace/MusicYOLO/predictions.json')

# Define confidence thresholds
#conf_thresholds = np.linspace(0.5, 0.95, 10)
conf_thresholds = [0.5]

# Compute average AP50
average_ap50, average_recall = average_precision(ground_truths, predictions, conf_thresholds)
print(f'Average AP50: {average_ap50}')
print(f'Average recall: {average_recall}')