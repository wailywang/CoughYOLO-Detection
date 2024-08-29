#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import time
from loguru import logger

import cv2
import json
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("MusicYOLO Evaluation")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--device", default="cpu", type=str, help="device to run model, either cpu or gpu")
    parser.add_argument("--conf", default=0.01, type=float, help="test confidence threshold")
    parser.add_argument("--nms", default=0.65, type=float, help="test NMS threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test image size")
    parser.add_argument("--coco_path", default=None, type=str, help="path to COCO ground truth json file")
    parser.add_argument("--image_dir", default=None, type=str, help="directory containing validation images")
    parser.add_argument("--output", default=None, type=str, help="path to save prediction results")
    return parser

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

def evaluate(predictor, image_dir, coco, confthre, output_path=None):
    image_list = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if os.path.splitext(img)[1] in IMAGE_EXT]
    predictions = []

    # 创建文件名到图像ID的映射
    filename_to_id = {img['file_name']: img['id'] for img in coco.dataset['images']}

    for image_path in image_list:
        filename = os.path.basename(image_path)
        if filename not in filename_to_id:
            logger.warning(f"Image {filename} not found in COCO dataset")
            continue
        
        img_id = filename_to_id[filename]  # 获取对应的image_id
        outputs, img_info = predictor.inference(image_path)
        
        if outputs[0] is None:
            continue

        bboxes = outputs[0][:, 0:4]
        bboxes /= img_info["ratio"]

        cls = outputs[0][:, 6]
        scores = outputs[0][:, 4] * outputs[0][:, 5]

        for i in range(len(bboxes)):
            if scores[i] < confthre:
                continue

            bbox = bboxes[i]
            prediction = {
                "image_id": img_id,  # 确保image_id对应正确
                "category_id": int(cls[i]),
                "bbox": [bbox[0].item(), bbox[1].item(), bbox[2].item() - bbox[0].item(), bbox[3].item() - bbox[1].item()],
                "score": scores[i].item()
            }
            predictions.append(prediction)
    
    # 保存预测结果到文件
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(predictions, f)

    coco_dt = coco.loadRes(predictions)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main(exp, args):
    # Load experiment and model
    model = exp.get_model()
    model.eval()

    if args.device == "gpu":
        model.cuda()

    logger.info("loading checkpoint")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    # Create predictor
    predictor = Predictor(model, exp, COCO_CLASSES, args.device)
    
    # Load COCO ground truth
    coco = COCO(args.coco_path)

    # Run evaluation
    evaluate(predictor, args.image_dir, coco, args.conf, args.output)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)
    main(exp, args)
