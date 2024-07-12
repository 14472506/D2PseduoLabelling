import os
import json
import cv2
import pickle
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pseudo_labeling.engine.trainer import PseudoTrainer
from pseudo_labeling.config import add_pseudo_config
from pseudo_labeling.modelling.my_rcnn import MyGeneralizedRCNN

def register_dataset():
    register_coco_instances("summerschool_test", {}, "datasets/summer_school_data/labelled/test/test_annotations.json", "datasets/summer_school_data/labelled/test/")

def setup_predictor():
    cfg = get_cfg()
    add_pseudo_config()
    cfg.merge_from_file("configs/pseudo_labeling/config_files/test_2.yaml")
    cfg.MODEL.WEIGHTS = "outputs/baseline/TEST_1/best_model.pth"
    cfg.DATASETS.TEST = ("my_dataset",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor

def make_predictions(predictor):
    dataset_dicts = DatasetCatalog.get("summerschool_test")
    outputs = []
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        outputs.append(predictor(img))

    with open("predictions.pkl", "wb") as f:
        pickle.dump(outputs, f)
    return outputs, dataset_dicts

def convert_to_coco_format(outputs, dataset_dicts):
    coco_results = []
    for i, output in enumerate(outputs):
        instances = output["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        for box, score, cls in zip(boxes, scores, classes):
            coco_results.append({
                "image_id": dataset_dicts[i]["image_id"],
                "category_id": cls + 1,  # COCO category ids start from 1
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "score": score
            })

    with open("coco_predictions.json", "w") as f:
        json.dump(coco_results, f, indent=4)

def evaluate_predictions():
    coco_gt = COCO("datasets/summer_school_data/labelled/test/test_annotations.json")
    coco_dt = coco_gt.loadRes("coco_predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    # Register dataset
    register_dataset()
    
    # Setup predictor
    predictor = setup_predictor()
    
    # Make predictions
    outputs, dataset_dicts = make_predictions(predictor)
    
    # Convert predictions to COCO format
    convert_to_coco_format(outputs, dataset_dicts)
    
    # Evaluate predictions
    evaluate_predictions()
