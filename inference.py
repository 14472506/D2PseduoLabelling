"""
Detials: Draft inference
"""
# imports =================================================
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os

# import common
import matplotlib.pyplot as plt
import cv2

# d2 utils
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# get jersey test registration
from pseudo_labeling.engine.trainer import PseudoTrainer
from pseudo_labeling.config import add_pseudo_config
from pseudo_labeling.modelling.my_rcnn import MyGeneralizedRCNN
from pseudo_labeling.modelling import mask_head 
from pseudo_labeling.modelling import custom_roi

from detectron2.data.datasets import register_coco_instances

# functions ===============================================
def setup(config_path, weights_path):
    cfg = get_cfg()
    add_pseudo_config(cfg)
    cfg.merge_from_file(config_path)    
    cfg.MODEL.WEIGHTS = weights_path 
    return(cfg)

def register_dataset():
    register_coco_instances("summerschool_test", {}, "datasets/summer_school_data/labelled/test/test_annotations.json", "datasets/summer_school_data/labelled/test/")

def main(config_path, weights_path, images_dir, outputs_dir):
    # get config and predictor
    register_dataset()
    cfg = setup(config_path, weights_path)
    predictor = DefaultPredictor(cfg)

    count = 0
    for im_path in os.listdir(images_dir):
        # skip json
        if im_path.endswith(".json"):
            continue
        if im_path.endswith(".txt"):
            continue
        
        # get prediction
        image = cv2.imread(os.path.join(images_dir, im_path))
        output = predictor(image)

        # visualise
        v = Visualizer(image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        out_image = v.get_image()[:,:,::-1]
        cv2.imwrite(os.path.join(outputs_dir, im_path), out_image)
        
        #plt.figure()
        #plt.imshow(cv2.cvtColor(v.get_image()[:,:,::-1], cv2.COLOR_BGR2RGB))
        #plt.savefig(os.path.join(outputs_dir, im_path))
        count += 1
        if count > 19:
            break
        
# execute =================================================
if __name__ == "__main__":
    main(
        "configs/pseudo_labeling/config_files/test_2.yaml",
        "outputs/ps_dev_testing/pseudo_labeling/100_epoch_bs8/best_model.pth",
        "datasets/jr_v5_unlabeled_data",
        "inference_out/pseudo_labeling_dif_4"
    )