import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_iou(box1, box2):
    x1, _, w1, _ = box1
    x2, _, w2, _ = box2

    xi1 = max(x1, x2)
    xi2 = min(x1 + w1, x2 + w2)

    inter_length = max(0, xi2 - xi1)
    box1_length = w1
    box2_length = w2
    union_length = box1_length + box2_length - inter_length

    return inter_length / union_length

def compute_ap(ground_truths, predictions, iou_threshold=0.5):
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

def evaluate_thresholds(ground_truths, predictions, conf_thresholds, iou_thresholds):
    results = []
    for conf_threshold in conf_thresholds:
        for iou_threshold in iou_thresholds:
            filtered_predictions = [pred for pred in predictions if pred['score'] >= conf_threshold]
            precision, recall = compute_ap(ground_truths, filtered_predictions, iou_threshold)
            results.append((conf_threshold, iou_threshold, precision, recall))
    return results

def plot_pr_curve(results):
    plt.figure(figsize=(10, 8))
    for conf_threshold, iou_threshold, precision, recall in results:
        plt.plot(recall, precision, marker='o', label=f'Conf: {conf_threshold}, IoU: {iou_threshold}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig('/mnt/workspace/MusicYOLO/prc_line.png', bbox_inches='tight')
    plt.show()

ground_truths = load_json('/mnt/workspace/MusicYOLO/datasets/images/valid/_annotations.coco.json')
predictions = load_json('/mnt/workspace/MusicYOLO/predictions.json')

conf_thresholds = np.linspace(0.5, 0.8, 10)
iou_thresholds = np.linspace(0.5, 0.8, 10)

results = evaluate_thresholds(ground_truths, predictions, conf_thresholds, iou_thresholds)

plot_pr_curve(results)