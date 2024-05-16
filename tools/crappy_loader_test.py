"""
Details
"""
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
import cv2
import os

from pseudo_labeling.data.build import build_pseudo_train_loader
from pseudo_labeling.data.registration import register_unlabeled, register_jersey_train

def main():
    """
    Details
    """
    register_jersey_train()
    register_unlabeled()
    print("registered_loaders")

    cfg = get_cfg()
    add_my_config(cfg)
    cfg.DATASETS.TRAIN = ("jersey_royal_train",)
    cfg.DATALOADER.NUM_WORKERS = 0
    print("config defined")

    data_loader = build_pseudo_train_loader(cfg)
    print("loader loaded")

    labeled_loader = data_loader[0]
    unlabeled_loader = data_loader[1]

    for idx, data in enumerate(labeled_loader):
        img = data[0]["image"].numpy().transpose(1, 2, 0)
        print(idx)
        print(img.shape)
    
def add_my_config(cfg):
    """
    Detials
    """
    _C = cfg
    _C.PSEUDO_LABELING = CN()
    _C.PSEUDO_LABELING.DATASET = ("unlabeled_data",)

if __name__ == "__main__":
    main()