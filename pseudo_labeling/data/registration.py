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
                            "datasets/jr_v5_unlabeled_data/unlabeled.json",
                            "datasets/jr_v5_unlabeled_data"
                            )

def register_jersey_train():
    """
    Detials
    """
    register_coco_instances("jersey_royal_train", {}, 
                                "datasets/jr_ds_v5/train/train.json",
                                "datasets/jr_ds_v5/train")
    
def register_jersey_val():
    """
    Detials
    """
    register_coco_instances("jersey_royal_val", {}, 
                                "datasets/jr_ds_v5/val/val.json",
                                "datasets/jr_ds_v5/val/val")

def register_jersey_test():
    """
    Detials
    """
    register_coco_instances("jersey_royal_test", {}, 
                                "datasets/jr_ds_v5/test/test.json",
                                "datasets/jr_ds_v5/test")