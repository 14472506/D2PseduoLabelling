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
                            "datasets/unlabeled_dataset/unlabeled.json",
                            "datasets/unlabeled_dataset"
                            )

def register_jersey_train():
    """
    Detials
    """
    register_coco_instances("jersey_royal_train", {}, 
                                "datasets/jersey_dataset_v4/train/train.json",
                                "datasets/jersey_dataset_v4/train")
    
def register_jersey_val():
    """
    Detials
    """
    register_coco_instances("jersey_royal_val", {}, 
                                "datasets/jersey_dataset_v4/val/val.json",
                                "datasets/jersey_dataset_v4/val")

def register_jersey_test():
    """
    Detials
    """
    register_coco_instances("jersey_royal_test", {}, 
                                "datasets/jersey_dataset_v4/test/test.json",
                                "datasets/jersey_dataset_v4/test")