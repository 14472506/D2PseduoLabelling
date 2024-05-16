"""
details
"""
# imports
# base
import copy
import logging
import os
from typing import List, Union
from PIL import Image

# third party
import numpy as np

# detectron2 and torch
import torch
import detectron2.data.detection_utils as utils
from detectron2.data import transforms as T
from detectron2.config import configurable

from pseudo_labeling.data.augmentations import build_strong_augmentation

# class
class UnlabeledDatasetMapper:
    """ 
    Detail
    """
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
    ):
        """
        Details
        """
        #fmt: off?
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.strong_augmentation    = build_strong_augmentation(is_train)
        self.image_format           = image_format
        # fmt: on?
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        student_dict = copy.deepcopy(dataset_dict)

        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        auged_image = aug_input.image
        image_pil = Image.fromarray(auged_image.astype("uint8"), "RGB")
        strong_auged_image = np.array(self.strong_augmentation(image_pil))
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(auged_image.transpose(2, 0, 1)))
        student_dict["strong_image"] = torch.as_tensor(np.ascontiguousarray(strong_auged_image.transpose(2, 0, 1)))

        return dataset_dict, student_dict

#def build_unlabeled_loader(cfg, mapper):
    """
    Details
    """
    