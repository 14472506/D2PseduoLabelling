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

def my_coco_train(train_percentage):
    """
    Detials
    """

    if train_percentage == 5:
        json = "datasets/coco2017/annotations/instances_train2017_05perc.json"
    elif train_percentage == 10:
        json = "datasets/coco2017/annotations/instances_train2017_10perc.json"
    elif train_percentage == 20:
        json = "datasets/coco2017/annotations/instances_train2017_20perc.json"
    else:
        json = "datasets/coco2017/annotations/instances_train2017.json"
    
    print("####################################################################")
    register_coco_instances("my_coco_train", {}, 
                                json,
                                "datasets/coco2017/images/train2017")
    register_coco_instances("my_coco_val", {}, 
                                "datasets/coco2017/annotations/instances_val2017.json",
                                "datasets/coco2017/images/val2017")

def register_jersey_train(train_percentage):
    """
    Detials
    """

    if train_percentage == 5:
        json = "datasets/jr_ds_v6/annotations/v6.1_train_05.json"
    elif train_percentage == 10:
        json = "datasets/jr_ds_v6/annotations/v6.1_train_10.json"
    elif train_percentage == 20:
        json = "datasets/jr_ds_v6/annotations/v6.1_train_20.json"
    else:
        json = "datasets/jr_ds_v6/annotations/v6.1_train.json"

    register_coco_instances("jersey_train", {}, 
                                json,
                                "datasets/jr_ds_v6/images")
    register_coco_instances("jersey_val", {}, 
                                "datasets/jr_ds_v6/annotations/v6.1_val.json",
                                "datasets/jr_ds_v6/images")
    
def register_jersey_val():
    """
    Detials
    """
    register_coco_instances("jersey_val", {}, 
                                "datasets/jr_ds_v6/annotations/v6.1_val.json",
                                "datasets/jr_ds_v6/images")

def register_jersey_test():
    """
    Detials
    """
    register_coco_instances("jersey_test", {}, 
                                "datasets/jr_ds_v6/annotations/v6.1_test.json",
                                "datasets/jr_ds_v6/images")