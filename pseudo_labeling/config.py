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
    _C.PSEUDO_LABELING.DATASET = ("",)
    _C.PSEUDO_LABELING.BURN_IN_ITERS = 0
    _C.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ = 20
    _C.PSEUDO_LABELING.EMA_KEEP_RATE = 0.998
    _C.PSEUDO_LABELING.METRIC_THRESHOLD = 1.0
    _C.PSEUDO_LABELING.CLASS_CONFIDENCE_THRESHOLD = 1.0