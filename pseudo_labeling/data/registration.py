"""
Detials
"""
# imports
from detectron2.data.datasets import register_coco_instances

# functions
def register_unlabeled():
    """
    Details
    """
    register_coco_instances("unlabeled_data", {},
                            "datasets/summer_school_data/unlabelled/unlabeled.json",
                            "datasets/summer_school_data/unlabelled"
                            )

def register_jersey_train():
    """
    Detials
    """
    register_coco_instances("summerschool_train", {}, 
                                "datasets/summer_school_data/labelled/train/images/train_annotations.json",
                                "datasets/summer_school_data/labelled/train/images")
    register_coco_instances("summerschool_val", {}, 
                                "datasets/summer_school_data/labelled/val/images/val_annotations.json",
                                "datasets/summer_school_data/labelled/val/images")
    
    
def register_jersey_val():
    """
    Detials
    """
    register_coco_instances("summerschool_val", {}, 
                                "datasets/summer_school_data/labelled/val/images/val_annotations.json",
                                "datasets/summer_school_data/labelled/val/images")

def register_jersey_test():
    """
    Detials
    """
    register_coco_instances("jersey_royal_test", {}, 
                                "datasets/jr_ds_v5/test/test.json",
                                "datasets/jr_ds_v5/test")