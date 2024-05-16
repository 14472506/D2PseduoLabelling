"""
Detials
"""
from detectron2.config import CfgNode as CN

# config adjust
def add_pseudo_config(cfg):
    """
    Detials
    """
    _C = cfg
    _C.SOLVER.OPTIMIZER = "SGD"

    _C.PSEUDO_LABELING = CN()
    _C.PSEUDO_LABELING.DATASET = ("unlabeled_data",)
    _C.PSEUDO_LABELING.BURN_IN_ITERS = 0
    _C.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ = 20
    _C.PSEUDO_LABELING.EMA_KEEP_RATE = 0.998