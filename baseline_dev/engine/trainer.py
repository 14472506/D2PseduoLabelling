"""
Detials
"""
# imports
# other
import os
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules

# detectron2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.engine import DefaultTrainer, TrainerBase, SimpleTrainer
from detectron2.engine.train_loop import AMPTrainer
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.engine import hooks
from detectron2.data import(
    build_detection_train_loader,
    build_detection_test_loader
    )
from detectron2.data import MetadataCatalog

# baseline dev trainer
from baseline_dev.data.registration import (
    register_jersey_train,
    register_jersey_test, 
    register_jersey_val
    )
from baseline_dev.engine.hooks import EvalHook
from baseline_dev.solver.optimizers import build_optimizer

# my trainer
class MyTrainer(DefaultTrainer):
    """
    Detials
    """
    def __init__(self, cfg):
        """
        Im compying this from unbiased teacher default setup and playing around with it
        """
        # configure world size and training parameters
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size)
        # configure model, optimiser, and data_loader
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # handling world size and the sorts
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffer=False
            )

        # initialising trainer base
        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with type {}".format(
                    dataset_name, evaluator_type
                )
            )
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Experimenting with custom data train loader registration here
        """
        register_jersey_train()
        return build_detection_train_loader(cfg)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Experimenting with custom data test loader registration here
        """
        if dataset_name == "jersey_royal_test":
            register_jersey_test()
        elif dataset_name == "jersey_royal_val":
            register_jersey_val()
        else:
            pass
        return build_detection_test_loader(cfg, dataset_name)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    def build_hooks(self):
        """
        building custom hooks for test
        """
        # all taken from detectron2, look into this
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        # inialise hooks list
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        # from here on is mine
        dataset_name = cfg.DATASETS.TEST[0]
        self.val_loader = self.build_test_loader(cfg, dataset_name)
        self.validation_evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
        def eval_function():
            return inference_on_dataset(self.model, self.val_loader, self.validation_evaluator)
        ret.append(EvalHook(cfg.TEST.EVAL_PERIOD, eval_function, self.checkpointer))
        
        # back to other 
        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret