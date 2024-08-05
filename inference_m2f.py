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
from detectron2.data.datasets import register_coco_instances

from pseudo_labeling.engine.trainer import PseudoTrainer
from mask2former.config import add_maskformer2_config
from pseudo_labeling.config import add_pseudo_config
from mask2former import MaskFormer

# functions ===============================================
def setup(config_path, weights_path):
    """ Initialise config and ammend based on command line arguments """
    cfg = get_cfg()
    # adding argments to base config to accomodate pseudo labeling
    add_maskformer2_config(cfg)
    add_pseudo_config(cfg)
    cfg.merge_from_file(config_path)  
    cfg.MODEL.WEIGHTS = weights_path
    return cfg

def main(config_path, weights_path, images_dir, outputs_dir):
    # get config and predictor
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
        "configs/pseudo_labeling/config_files/ps_m2f.yaml",
        "outputs/m2f_test/baseline/pre_training_best_model.pth",
        "datasets/jr_v5_unlabeled_data",
        "inference_out/m2f_baseline"
    )