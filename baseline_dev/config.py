"""
Detials
"""
from detectron2.config import CfgNode as CN

# config adjust
def add_my_config(cfg):
    """
    Detials
    """
    _C = cfg
    _C.SOLVER.OPTIMIZER = "SGD"