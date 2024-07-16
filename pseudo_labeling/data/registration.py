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
    register_coco_instances("jersey_train", {}, 
                                "datasets/jr_ds_v6/annotatiosn/v6.1_train.json",
                                "datasets/jr_ds_v6/all_labelled")
    register_coco_instances("jersey_val", {}, 
                                "datasets/jr_ds_v6/annotatiosn/v6.1_val.json",
                                "datasets/jr_ds_v6/all_labelled")
    
    
def register_jersey_val():
    """
    Detials
    """
    register_coco_instances("jersey_val", {}, 
                                "datasets/jr_ds_v6/annotatiosn/v6.1_val.json",
                                "datasets/jr_ds_v6/all_labelled")

def register_jersey_test():
    """
    Detials
    """
    register_coco_instances("jersey_test", {}, 
                                "datasets/jr_ds_v6/annotatiosn/v6.1_test.json",
                                "datasets/jr_ds_v6/all_labelled")