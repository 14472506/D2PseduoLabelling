"""
Detials
"""
# imports
# other
import os
from collections import OrderedDict
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import logging
import time
import cv2
import copy

import torch

# detectron2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.engine import DefaultTrainer, TrainerBase, SimpleTrainer
from detectron2.engine.train_loop import AMPTrainer
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.engine import hooks
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog
from detectron2.utils.events import EventStorage
from detectron2.structures import PolygonMasks, Boxes, BoxMode, Instances

# pseudo labeling imports
from pseudo_labeling.data.build import build_pseudo_train_loader
from pseudo_labeling.data.registration import (
    register_jersey_train,
    register_unlabeled,
    register_jersey_val,
    register_jersey_test
    )
from pseudo_labeling.engine.hooks import EvalHook
from pseudo_labeling.solver.optimizers import build_optimizer

# classes 
class PseudoTrainer(DefaultTrainer):
    """
    Detials
    """
    # =========================================================================
    # Init
    # =========================================================================
    def __init__(self, cfg):
        """
        Detials
        """
        # configure world size and training parameters
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size)

        # get model, and optimiser
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # get model teacher
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        self.model_teacher.pseudo_labeling = True
        self.model_teacher.eval()

        # get data loader
        data_loader = self.build_train_loader(cfg)

        # handling world size
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids = [comm.get_local_rank()], broadcast_buffer=False
            )

        # initialise trainer base
        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else PseudoSimpleTrainer)(
            model, data_loader, optimizer
        )

        # TODO add ensembled model for checkpointing here
        
        # get optimiser and scheduler
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        # training loop parameters
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        # register hooks
        self.register_hooks(self.build_hooks())
    
    # =========================================================================
    # Data loader methods
    # =========================================================================
    @classmethod
    def build_train_loader(cls, cfg):
        """ 
        Detials
        """
        register_jersey_train()
        register_unlabeled()
        return build_pseudo_train_loader(cfg)
    
    @classmethod
    # THIS NEEDS CHANGING TO WORK WITH THE BASH FILES
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
        
    # =========================================================================
    # Optimiser
    # =========================================================================
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Detials
        """
        return build_optimizer(cfg, model)

    # =========================================================================
    # Build Hooks
    # =========================================================================
    def build_hooks(self):
        """
        Detials
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

        ### ADDED FOR DEV, WILL NEED TO BE RE ASSESSED
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

    # =========================================================================
    # Training
    # =========================================================================
    def train(self):
        """
        Detials
        """
        self.train_loop(self.start_iter, self.max_iter)

    def train_loop(self, start_iter, max_iter):
        """
        Details
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()

            except Exception:
                logger.exception("Exception during training:")
                raise

            finally:
                self.after_train()

    def run_step(self):
        """
        Details
        """
        # set training iter
        self._trainer.iter = self.iter

        # start time
        start = time.perf_counter()

        # get labeled data
        labeled_data = next(self._trainer._labeled_data_loader_iter)
        data_time = time.perf_counter() - start

        if self.iter < self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
            # forward pass on model
            loss_dict = self.model(labeled_data)
            losses = sum(loss_dict.values())
        else:
            # condition based teacher update
            if self.iter == self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                self._update_teacher_model(keep_rate=0.00)
            elif(self.iter - self.cfg.PSEUDO_LABELING.BURN_IN_ITERS) % self.cfg.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ == 0:
                self._update_teacher_model(keep_rate = self.cfg.PSEUDO_LABELING.EMA_KEEP_RATE)

            # get pseduo labeled data
            pseudo_labeled_data = self.pseudo_label()

            # forward pass of on labeled and unlabeled data
            record_dict = {}
            labeled_loss_dict = self.model(labeled_data)
            unlabeled_loss_dict = self.model(pseudo_labeled_data)
            record_dict["labeled"] = labeled_loss_dict
            record_dict["unlabeled"] = unlabeled_loss_dict

            # process losses
            loss_dict = {}

            for key in record_dict.keys():
                # simple summing for now but more complex weighting and approaches can be used here.
                loss_dict[key] = sum(record_dict[key].values())
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()
        #self.after_backward()
        self.optimizer.step()

    # =========================================================================
    # Pseudo Labeling
    # =========================================================================
    def pseudo_label(self):
        """
        Details
        """    
        got_pseudo_label = False
        while not got_pseudo_label:
            # Get predictions from unlabeled data
            unlabeled_data = next(self._trainer._unlabeled_data_loader_iter)

            # Assuming the batch size is greater than 1
            student_images = [data[1]["strong_image"] for data in unlabeled_data]
            unlabeled_data = [data[0] for data in unlabeled_data]

            batch_preds = self.model_teacher(unlabeled_data)

            # Process predictions for each image in the batch
            for i, preds in enumerate(batch_preds):
                preds = preds["instances"].to("cpu")

                pred_scores = preds.scores.detach().numpy()
                pred_masks = preds.pred_masks.detach().numpy() / 255
                pred_boxes = preds.pred_boxes.tensor.detach().numpy()
                pred_classes = preds.pred_classes.detach().numpy()

                # Filter masks by prediction score
                cf_pred_scores = []
                cf_pred_masks = []
                cf_pred_boxes = []
                cf_pred_classes = []

                for j in range(len(pred_scores)):
                    if pred_scores[j] < self.cfg.PSEUDO_LABELING.CLASS_CONFIDENCE_THRESHOLD:
                        continue
                    cf_pred_scores.append(pred_scores[j])
                    cf_pred_masks.append(pred_masks[j])
                    cf_pred_boxes.append(pred_boxes[j])
                    cf_pred_classes.append(pred_classes[j])

                # Filter masks by metric
                mf_pred_scores = []
                mf_pred_masks = []
                mf_pred_boxes = []
                mf_pred_classes = []

                for j in range(len(cf_pred_scores)):
                    conf_score = cf_pred_scores[j]
                    mask = cf_pred_masks[j]

                    # Check mask
                    binary_mask = np.where(mask >= 0.5, 1, 0)
                    area = np.count_nonzero(binary_mask)

                    if area < 50:
                        continue

                    # Get volumetric symmetry metric
                    higher_volume = mask[mask >= 0.5] - 0.5
                    lower_volume = np.minimum(mask, 0.5)
                    vol_sym = conf_score * (higher_volume.sum() / lower_volume.sum()) ** 2

                    if vol_sym < self.cfg.PSEUDO_LABELING.METRIC_THRESHOLD:
                        continue

                    mf_pred_scores.append(cf_pred_scores[j])
                    mf_pred_masks.append(binary_mask)
                    mf_pred_boxes.append(cf_pred_boxes[j])
                    mf_pred_classes.append(cf_pred_classes[j])

                if len(mf_pred_scores) == 0:
                    continue

                # Data Post Processing
                raw_polygons = self.masks_to_polygone_masks(mf_pred_masks)
                all_polygons = []
                for polys in raw_polygons:
                    good_polys = []
                    for poly in polys:
                        if len(poly) > 4:
                            good_polys.append(poly)
                    all_polygons.append(good_polys)

                polygone_masks = PolygonMasks(all_polygons)
                mf_pred_boxes = np.array(mf_pred_boxes)
                boxes = Boxes(torch.tensor(mf_pred_boxes).float())
                instances = Instances((unlabeled_data[i]["height"], unlabeled_data[i]["width"]))
                instances.gt_boxes = boxes
                instances.gt_masks = polygone_masks
                instances.gt_classes = torch.tensor(mf_pred_classes)

                # Update unlabeled data with pseudo labels
                unlabeled_data[i]["image"] = student_images[i]
                unlabeled_data[i]["instances"] = instances

                got_pseudo_label = True
        
        # Return the batch of unlabeled data with pseudo labels
        return unlabeled_data

    def masks_to_polygone_masks(self, masks):
        """ 
        Details
        """
        all_polygons = []
        masks = np.array(masks)        
        for i in range(masks.shape[0]):
            contours, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [contour.flatten().tolist() for contour in contours]
            all_polygons.append(polygons)

        return all_polygons
    
    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        """
        Details
        """
        student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    # =========================================================================
    # Evaluator
    # =========================================================================
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
    
class PseudoSimpleTrainer(SimpleTrainer):
    """
    Details
    """
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        #gather_metric_period=1,
        #zero_grad_before_forward=False,
        #async_write_metrics=False,
    ):
        super().__init__(
            model,
            data_loader,
            optimizer,
            #gather_metric_period,
            #zero_grad_before_forward,
            #async_write_metrics,
        )

        self._labeled_data_loader_iter_obj = None
        self._unlabeled_data_loader_iter_obj = None

    @property
    def _labeled_data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._labeled_data_loader_iter_obj is None:
            self._labeled_data_loader_iter_obj = iter(self.data_loader[0])
        return self._labeled_data_loader_iter_obj
    
    @property
    def _unlabeled_data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._unlabeled_data_loader_iter_obj is None:
            self._unlabeled_data_loader_iter_obj = iter(self.data_loader[1])
        return self._unlabeled_data_loader_iter_obj
