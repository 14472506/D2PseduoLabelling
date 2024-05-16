"""
Detials
"""
# imports
from typing import List, Any, Dict, Set
import torch

# functions
def build_optimizer(cfg, model):
    """
    Detiails
    """
    def maybe_add_full_model_gradient_clipping(optim):
        """
        Details
        """
        pass
        # gradient clipping function required

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    # get params
    for key, value in model.named_parameters(recurse=True):
        # skip if parameter does not require grad
        if not value.requires_grad:
            continue
        # skip already existing to avoid duplication
        if value in memo:
            continue
        memo.add(value)

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # get optimizer
    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = torch.optim.Adam(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")

    # return optimizer
    return optimizer