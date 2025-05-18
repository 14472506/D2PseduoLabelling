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
import statistics
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import torch.nn.functional as TF

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
from detectron2.structures import PolygonMasks, Boxes, BoxMode, Instances, BitMasks
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer, ColorMode

from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY

# pseudo labeling imports
from pseudo_labeling.data.build import build_pseudo_train_loader
from pseudo_labeling.data.registration import (
    register_jersey_train,
    register_unlabeled,
    register_jersey_val,
    register_jersey_test,
    my_coco_train,
    register_train_cityscapes,
    register_val_cityscapes,
    register_test_cityscapes,
    register_unlabeled_cityscapes
    )
from pseudo_labeling.engine.hooks import EvalHook
from pseudo_labeling.solver.optimizers import build_optimizer
from pseudo_labeling.solver.losses import calculate_affinity_mask

from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper

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
        # Set requires_grad=False for teacher model parameters
        for param in self.model_teacher.parameters():
            param.requires_grad = False
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
        self._trainer = (AMPPseudoTrainer if cfg.SOLVER.AMP.ENABLED else PseudoSimpleTrainer)(
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
        self.teacher_checkpointer = DetectionCheckpointer(
            model_teacher,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        # training loop parameters
        self.start_iter = 0

        self.metric_thresh = cfg.PSEUDO_LABELING.METRIC_THRESHOLD
        self.metric_mean_acc = 0
        self.metric_mean_count = 0
        self.metric_mean_val = 0

        self.max_iter = cfg.SOLVER.MAX_ITER
        self.pre_training = cfg.PSEUDO_LABELING.PRE_TRAIN
        self.burn_in = cfg.PSEUDO_LABELING.BURN_IN
        self.cfg = cfg

        # register hooks
        self.register_hooks(self.build_hooks())


    # =========================================================================
    # Build Model Method
    # =========================================================================
    #def build_model(cls, cfg):
    #    model = build_model(cfg)
    #    model.roi_heads = MaskRCNNConvUpsampleHead(cfg, model.backbone.output_shape())
    #    return model
    
    # =========================================================================
    # Data loader methods
    # =========================================================================
    @classmethod
    def build_train_loader(cls, cfg):
        """ 
        Detials
        """
        if "jersey_train" in cfg.DATASETS.TRAIN:
            register_jersey_train(cfg.PSEUDO_LABELING.TRAIN_PERC)
            register_unlabeled()
        if "my_coco_train" in cfg.DATASETS.TRAIN:
            my_coco_train(cfg.PSEUDO_LABELING.TRAIN_PERC)
            register_unlabeled()
        if "cityscapes_train" in cfg.DATASETS.TRAIN:
            register_train_cityscapes(cfg.PSEUDO_LABELING.TRAIN_PERC)
            register_unlabeled_cityscapes()

        if cfg.INPUT.DATASET_MAPPER_NAME == "m2f":
            return build_pseudo_train_loader(cfg, labeled_mapper=COCOInstanceNewBaselineDatasetMapper(cfg, True )) 
        else:
            return build_pseudo_train_loader(cfg)
    
    @classmethod
    # THIS NEEDS CHANGING TO WORK WITH THE BASH FILES
    def build_test_loader(cls, cfg, dataset_name):
        """
        Experimenting with custom data test loader registration here
        """
        if dataset_name == "jersey_test":
            register_jersey_test()
        elif dataset_name == "jersey_royal_val":
            register_jersey_val()
        elif dataset_name == "cityscapes_test":
            register_test_cityscapes()
        elif dataset_name == "cityscapes_val":
            register_val_cityscapes()
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
        self.eval_hook = EvalHook(cfg, cfg.TEST.EVAL_PERIOD, eval_function, self.checkpointer, self.teacher_checkpointer)
        ret.append(self.eval_hook)
        
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

    ###############################################################################################
    # RUN STEP DEV
    def run_step(self):
        """
        Details
        """
        # Setup run step: get current iter and start timer
        self._trainer.iter = self.iter
        start = time.perf_counter()

        # Always collect labeled data
        labeled_data = next(self._trainer._labeled_data_loader_iter)
        data_time = time.perf_counter() - start

        # If in pre-train iteration stage only carry out supervised forward pass
        if self.pre_training and self.iter < self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
            loss_dict = self.model(labeled_data)
            losses = sum(loss_dict.values())

        # Else do pseudo labeling
        else:
            # If burn-in and in burn-in range
            if self.burn_in and self.iter < self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                # In first instance load student weights to teacher and initialize burn-in weights for student. Otherwise do nothing. Teacher is frozen.
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
                    print("INIT BURN IN STAGE, LOADING STUDENT WEIGHTS")
                    self._update_teacher_model(keep_rate=0.00)
                    # Load student weights
                    DetectionCheckpointer(self.model).load(self.cfg.PSEUDO_LABELING.BURN_IN_STUDENT_WEIGHTS)
            # If in distillation range with burn-in present
            elif self.burn_in and self.iter >= self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                # Distillation after burn-in, so load best student. Teacher is already in place, in the first instance, afterward carry out distillation with given EMA keep rate.
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                    # Load best student weights
                    DetectionCheckpointer(self.model).load(os.path.join(self.cfg.OUTPUT_DIR, "burn_in_best_model.pth"))
                elif (self.iter - self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS) % self.cfg.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ == 0:
                    if self.eval_hook.distillation_burn_in:
                        self._update_teacher_model(keep_rate=self.cfg.PSEUDO_LABELING.EMA_KEEP_RATE)
            # If there is no burn-in
            elif not self.burn_in:
                # In first instance load student weights to teacher and initialize burn-in weights for student. Otherwise do nothing. Teacher is frozen.
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
                    print("INIT DISTILLATION, LOADING STUDENT WEIGHTS")
                    self._update_teacher_model(keep_rate=0.00)
                    # Load student weights
                    DetectionCheckpointer(self.model).load(self.cfg.PSEUDO_LABELING.BURN_IN_STUDENT_WEIGHTS)
                elif (self.iter - self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS) % self.cfg.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ == 0:
                    if self.eval_hook.distillation_burn_in:
                        self._update_teacher_model(keep_rate=self.cfg.PSEUDO_LABELING.EMA_KEEP_RATE)

            # Get pseudo-labeled data
            if self.cfg.INPUT.DATASET_MAPPER_NAME == "m2f":
                pseudo_labeled_data = self.m2f_pseudo_label()
                #visualize_and_save_instances(pseudo_labeled_data)
            else:
                pseudo_labeled_data = self.mrcnn_pseudo_label()

            # Forward pass on labeled and unlabeled data
            record_dict = {}
            labeled_loss_dict = self.model(labeled_data)
            unlabeled_loss_dict = self.model(pseudo_labeled_data)
            record_dict["labeled"] = labeled_loss_dict
            record_dict["unlabeled"] = unlabeled_loss_dict

            # Process losses
            loss_dict = {}
            for key in record_dict.keys():
                # Weighting here later
                if key == "unlabeled":
                    #for loss_key in record_dict[key].keys():
                    #    if loss_key == "loss_mask":
                    #        record_dict[key][loss_key] = mean_met_val* record_dict[key][loss_key]
                    #loss_dict[key] = sum(record_dict[key].values())
                    #print(loss_dict[key], mean_met_val)
                    loss_dict[key] = self.cfg.PSEUDO_LABELING.LOSS_WEIGHTING * sum(record_dict[key].values())
                else:
                    loss_dict[key] = sum(record_dict[key].values())
                
            losses = sum(loss_dict.values())

        if self.cfg.PSEUDO_LABELING.METRIC_USE == "dynamic":
            if self.iter % self.cfg.TEST.EVAL_PERIOD == 0:
                if self.iter != 0:
                    self.metric_thresh = self.metric_mean_val - self.cfg.PSEUDO_LABELING.METRIC_OFFSET
                    print("### NEW_THRESH ######################################")
                    print(self.metric_thresh)
                    self.metric_mean_count = 0
                    self.metric_mean_acc = 0

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
    ###############################################################################################

    # =========================================================================
    # Pseudo Labeling
    # =========================================================================

    def mrcnn_pseudo_label(self):
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

            metric_values = []
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
                mf_pred_binary_masks = []
                #mf_pred_logit_masks = []
                mf_pred_boxes = []
                mf_pred_classes = []
                mf_vol_sym = []

                for j in range(len(cf_pred_scores)):
                    conf_score = cf_pred_scores[j]
                    mask = cf_pred_masks[j]

                    # Check mask
                    binary_mask = np.where(mask >= 0.5, 1, 0)
                    area = np.count_nonzero(binary_mask)
                    
                    if area < 50:
                        continue

                    ## Get volumetric symmetry metric - Logit_symmetry
                    higher_volume = mask[mask >= 0.5] - 0.5
                    lower_volume = np.minimum(mask, 0.5)
                    vol_sym = conf_score * (higher_volume.sum() / lower_volume.sum()) ** 2

                    # Other
                    #good_volume = mask[mask >= 0.5]
                    #bad_volume = mask
                    #vol_sym = conf_score * (good_volume.sum()/bad_volume.sum())**2

                    if vol_sym < self.metric_thresh:
                        continue

                    mf_pred_scores.append(cf_pred_scores[j])
                    mf_pred_binary_masks.append(binary_mask)
                    #mf_pred_logit_masks.append(mask)
                    mf_pred_boxes.append(cf_pred_boxes[j])
                    mf_pred_classes.append(cf_pred_classes[j])
                    mf_vol_sym.append(vol_sym)

                if len(mf_pred_scores) == 0:
                    continue

                #mf_pred_logit_masks_tensor = torch.tensor(mf_pred_binary_masks, dtype=torch.float32, device="cpu")
                #affinity_binary_masks = calculate_affinity_mask(mf_pred_logit_masks_tensor)

                # Prepare images, binary masks, and affinity masks for visualization
                #images = [unlabeled_data[i]["image"].permute(1, 2, 0).cpu().numpy()]
                #binary_masks = [np.array(mf_pred_binary_masks[i]) for i in range(len(mf_pred_binary_masks))]
                #affinity_masks = [affinity_binary_masks[i].cpu().numpy().squeeze() for i in range(len(affinity_binary_masks))]

                # Visualize the affinity masks
                #visualize_masks(images, binary_masks, affinity_masks, titles=[f"Image {i}" for i in range(len(images))], save_path='mask_visualization_affinity.png')

                # Data Post Processing
                raw_polygons = self.masks_to_polygone_masks(mf_pred_binary_masks)
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
                #instances.gt_masks_logits = torch.tensor(mf_pred_binary_masks)
                instances.gt_classes = torch.tensor(mf_pred_classes)
                instances.gt_metric_score = torch.tensor(mf_vol_sym)

                # Update unlabeled data with pseudo labels
                unlabeled_data[i]["image"] = student_images[i]
                unlabeled_data[i]["instances"] = instances
                metric_values.extend(mf_vol_sym)

                got_pseudo_label = True
        
        self.metric_mean_count += 1
        self.metric_mean_acc += statistics.mean(metric_values)
        self.metric_mean_val = self.metric_mean_acc/self.metric_mean_count
        # Return the batch of unlabeled data with pseudo labels

        return unlabeled_data

    def m2f_pseudo_label(self):
        """
        Generate pseudo labels for the unlabeled data using the teacher model.
        """
        got_pseudo_label = False
        while not got_pseudo_label:
            # Get predictions from unlabeled data
            unlabeled_data = next(self._trainer._unlabeled_data_loader_iter)

            # Assuming the batch size is greater than 1
            student_images = [data[1]["strong_image"] for data in unlabeled_data]
            unlabeled_data = [data[0] for data in unlabeled_data]

            batch_preds = self.model_teacher(unlabeled_data)

            metric_values = []
            # Process predictions for each image in the batch
            skip_batch = False 
            for i, preds in enumerate(batch_preds):
                preds = preds["instances"].to("cpu")

                pred_scores = preds.scores
                pred_masks = preds.pred_masks
                pred_boxes = preds.pred_boxes.tensor
                pred_classes = preds.pred_classes

                # Filter masks by prediction score
                mask_filter = pred_scores >= self.cfg.PSEUDO_LABELING.CLASS_CONFIDENCE_THRESHOLD
                filtered_scores = pred_scores[mask_filter]
                filtered_masks = pred_masks[mask_filter]
                filtered_boxes = pred_boxes[mask_filter]
                filtered_classes = pred_classes[mask_filter]

                # Filter masks by metric
                valid_indices = []
                mf_vol_sym = []

                for j in range(filtered_scores.size(0)):
                    mask = filtered_masks[j]

                    # Check mask area
                    binary_mask = (mask >= 0.5).float()
                    area = binary_mask.sum().item()

                    #if area < 50:
                    #    continue

                    # Calculate volumetric symmetry metric
                    higher_volume = mask[mask >= 0.5] - 0.5
                    lower_volume = torch.clamp(mask, max=0.5)
                    vol_sym = filtered_scores[j] * (higher_volume.sum() / lower_volume.sum()) ** 2

                    if vol_sym < self.metric_thresh:
                        continue
                        
                    valid_indices.append(j)
                    mf_vol_sym.append(vol_sym.item())

                if len(valid_indices) == 0:
                    skip_batch = True
                    break

                # Data Post Processing
                valid_indices = torch.tensor(valid_indices, dtype=torch.long)
                selected_masks = filtered_masks[valid_indices]
                selected_boxes = filtered_boxes[valid_indices]
                selected_classes = filtered_classes[valid_indices]

                # Resize masks to fit the image size
                image_shape = (student_images[i].shape[1], student_images[i].shape[2])
                resized_masks = TF.interpolate(selected_masks.unsqueeze(1).float(), size=image_shape, mode='nearest').squeeze(1)
                resized_masks = (resized_masks >= 0.5).float()

                # Create BitMasks from the resized masks tensor
                bitmask_masks = BitMasks(resized_masks)

                # Create Instances object
                instances = Instances(image_shape)
                instances.gt_boxes = Boxes(selected_boxes)
                instances.gt_masks = bitmask_masks
                instances.gt_classes = selected_classes
                instances.gt_metric_score = torch.tensor(mf_vol_sym)

                # Update unlabeled data with pseudo labels
                unlabeled_data[i]["image"] = student_images[i]
                unlabeled_data[i]["instances"] = instances
                metric_values.extend(mf_vol_sym)

            if skip_batch:
                print("Skipping batch due to no valid pseudo labels.")
                continue

            got_pseudo_label = True

        self.metric_mean_count += 1
        self.metric_mean_acc += statistics.mean(metric_values)
        self.metric_mean_val = self.metric_mean_acc / self.metric_mean_count

        # Return the batch of unlabeled data with pseudo labels
        return unlabeled_data

    #def prepare_targets(self, targets, images):
    #    h_pad, w_pad = images.tensor.shape[-2:]
    #    new_targets = []
    #    for targets_per_image in targets:
    #        if isinstance(targets_per_image.gt_masks, BitMasks):
    #            gt_masks_tensor = targets_per_image.gt_masks.tensor
    #            (f"gt_masks_tensor type: {type(gt_masks_tensor)}")
    #            print(f"gt_masks_tensor shape: {gt_masks_tensor.shape}")
    #        else:
    #            raise ValueError("Expected gt_masks to be of type BitMasks")
    #
    #        padded_masks = torch.zeros((gt_masks_tensor.shape[0], h_pad, w_pad), dtype=gt_masks_tensor.dtype, device=gt_masks_tensor.device)
    #        padded_masks[:, : gt_masks_tensor.shape[1], : gt_masks_tensor.shape[2]] = gt_masks_tensor
    #        new_targets.append(
    #            {
    #                "labels": targets_per_image.gt_classes,
    #                "masks": padded_masks,
    #            }
    #        )
    #    return new_targets
    
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
    def _update_teacher_model(self, keep_rate=0.9996):
        """
        Details
        """
        student_model_dict = self.model.state_dict()
        teacher_model_dict = self.model_teacher.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in teacher_model_dict.items():
            if key in student_model_dict.keys():
                # THIS IS HACKY AS HELL BUT IT SEEMS TO WORK SO DONT QUESTION IT
                if "pixel_decoder" in key:
                    # Modify key to remove pixel_decoder
                    mod_key = key.replace("pixel_decoder.", "")
                    new_teacher_dict[mod_key] = (student_model_dict[key] * (1 - keep_rate) + value * keep_rate)
                else:
                    new_teacher_dict[key] = (student_model_dict[key] * (1 - keep_rate) + value * keep_rate)
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
    
def visualize_masks(images, binary_masks, affinity_masks, titles=None, save_path='mask_visualization_affinity.png'):
    """
    Visualize a batch of images with their corresponding binary masks and affinity masks.

    Args:
        images (list or numpy array): List or array of images to visualize.
        binary_masks (list or numpy array): List or array of binary masks to visualize.
        affinity_masks (list or numpy array): List or array of affinity masks to visualize.
        titles (list): Optional list of titles for each subplot.
        save_path (str): Path to save the visualization image.
    """
    num_images = len(images)

    plt.figure(figsize=(15, 5 * num_images))

    for i in range(num_images):
        plt.subplot(num_images, 3, 3 * i + 1)
        plt.imshow(images[i], cmap='gray')
        if titles:
            plt.title(f"{titles[i]} Image")
        plt.axis('off')

        plt.subplot(num_images, 3, 3 * i + 2)
        binary_mask = binary_masks[i]
        if isinstance(binary_mask, torch.Tensor):
            binary_mask = binary_mask.cpu().numpy()
        if binary_mask.ndim == 3 and binary_mask.shape[2] == 1:  # Handle case where masks have shape [H, W, 1]
            binary_mask = binary_mask.squeeze(-1)
        binary_mask = (binary_mask - binary_mask.min()) / (binary_mask.max() - binary_mask.min())
        plt.imshow(binary_mask, cmap='gray')
        if titles:
            plt.title(f"{titles[i]} Binary Mask")
        plt.axis('off')

        plt.subplot(num_images, 3, 3 * i + 3)
        affinity_mask = affinity_masks[i]
        if isinstance(affinity_mask, torch.Tensor):
            affinity_mask = affinity_mask.cpu().numpy()
        if affinity_mask.ndim == 3 and affinity_mask.shape[2] == 1:  # Handle case where masks have shape [H, W, 1]
            affinity_mask = affinity_mask.squeeze(-1)
        affinity_mask = (affinity_mask - affinity_mask.min()) / (affinity_mask.max() - affinity_mask.min())
        plt.imshow(affinity_mask, cmap='gray')
        if titles:
            plt.title(f"{titles[i]} Affinity Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

class GuidedDistPseudoLabelling(PseudoTrainer):
    """
    Detials
    """
    def m2f_pseudo_label(self):
        """
        Generate pseudo labels for the unlabeled data using the teacher model.
        """
        got_pseudo_label = False
        while not got_pseudo_label:
            # Get predictions from unlabeled data
            unlabeled_data = next(self._trainer._unlabeled_data_loader_iter)

            # Assuming the batch size is greater than 1
            student_images = [data[1]["strong_image"] for data in unlabeled_data]
            unlabeled_data = [data[0] for data in unlabeled_data]

            batch_preds = self.model_teacher(unlabeled_data)

            metric_values = []
            skip_batch = False
            for i, preds in enumerate(batch_preds):
                preds = preds["instances"].to("cpu")
                pred_scores = preds.scores
                pred_masks = preds.pred_masks
                pred_boxes = preds.pred_boxes.tensor
                pred_classes = preds.pred_classes
                # Filter masks by prediction score
                mask_filter = pred_scores >= self.cfg.PSEUDO_LABELING.CLASS_CONFIDENCE_THRESHOLD
                filtered_scores = pred_scores[mask_filter]
                filtered_masks = pred_masks[mask_filter]
                filtered_boxes = pred_boxes[mask_filter]
                filtered_classes = pred_classes[mask_filter]

                # Filter masks based on sum of logits
                valid_indices = []
                summed_logits = []

                for j in range(filtered_scores.size(0)):
                    mask = filtered_masks[j]

                    # Sum of logits
                    logits_sum = mask.sum().item()
                    if logits_sum < self.metric_thresh:
                        continue

                    valid_indices.append(j)
                    summed_logits.append(logits_sum)

                if len(valid_indices) == 0:
                    skip_batch = True
                    break

                # Data Post Processing
                valid_indices = torch.tensor(valid_indices, dtype=torch.long)
                selected_masks = filtered_masks[valid_indices]
                selected_boxes = filtered_boxes[valid_indices]
                selected_classes = filtered_classes[valid_indices]

                # Resize masks to fit the image size
                image_shape = (student_images[i].shape[1], student_images[i].shape[2])
                resized_masks = TF.interpolate(
                    selected_masks.unsqueeze(1).float(), size=image_shape, mode='nearest'
                ).squeeze(1)
                resized_masks = (resized_masks >= 0.5).float()

                # Create BitMasks from the resized masks tensor
                bitmask_masks = BitMasks(resized_masks)

                # Create Instances object
                instances = Instances(image_shape)
                instances.gt_boxes = Boxes(selected_boxes)
                instances.gt_masks = bitmask_masks
                instances.gt_classes = selected_classes
                instances.gt_metric_score = torch.tensor(summed_logits)

                # Update unlabeled data with pseudo labels
                unlabeled_data[i]["image"] = student_images[i]
                unlabeled_data[i]["instances"] = instances
                metric_values.extend(summed_logits)

            if skip_batch:
                print("Skipping batch due to no valid pseudo labels.")
                continue

            got_pseudo_label = True

        self.metric_mean_count += 1
        self.metric_mean_acc += statistics.mean(metric_values)
        self.metric_mean_val = self.metric_mean_acc / self.metric_mean_count

        # Return the batch of unlabeled data with pseudo labels
        return unlabeled_data

class AMPPseudoTrainer(PseudoSimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
    ):
        """
        Args:
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward,
                async_write_metrics: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
            precision: torch.dtype as the target precision to cast to in computations
        """
        #unsupported = "AMPTrainer does not support single-process multi-device training!"
        #if isinstance(model, DistributedDataParallel):
        #    assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        #assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer)


        from torch.cuda.amp import GradScaler
        grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        # Setup run step: get current iter and start timer
        self._trainer.iter = self.iter
        start = time.perf_counter()
        
        # Always collect labeled data
        labeled_data = next(self._trainer._labeled_data_loader_iter)
        data_time = time.perf_counter() - start        
        
        # If in pre-train iteration stage only carry out supervised forward pass
        if self.pre_training and self.iter < self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
            loss_dict = self.model(labeled_data)
            losses = sum(loss_dict.values())
        # Else do pseudo labeling
        else:
            # If burn-in and in burn-in range
            if self.burn_in and self.iter < self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                # In first instance load student weights to teacher and initialize burn-in weights for student. Otherwise do nothing. Teacher is frozen.
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
                    print("INIT BURN IN STAGE, LOADING STUDENT WEIGHTS")
                    self._update_teacher_model(keep_rate=0.00)
                    # Load student weights
                    DetectionCheckpointer(self.model).load(self.cfg.PSEUDO_LABELING.BURN_IN_STUDENT_WEIGHTS)
            # If in distillation range with burn-in present
            elif self.burn_in and self.iter >= self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                # Distillation after burn-in, so load best student. Teacher is already in place, in the first instance, afterward carry out distillation with given EMA keep rate.
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                    # Load best student weights
                    DetectionCheckpointer(self.model).load(os.path.join(self.cfg.OUTPUT_DIR, "burn_in_best_model.pth"))
                elif (self.iter - self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS) % self.cfg.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ == 0:
                    if self.eval_hook.distillation_burn_in:
                        self._update_teacher_model(keep_rate=self.cfg.PSEUDO_LABELING.EMA_KEEP_RATE)
            # If there is no burn-in
            elif not self.burn_in:
                # In first instance load student weights to teacher and initialize burn-in weights for student. Otherwise do nothing. Teacher is frozen.
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
                    print("INIT DISTILLATION, LOADING STUDENT WEIGHTS")
                    self._update_teacher_model(keep_rate=0.00)
                    # Load student weights
                    DetectionCheckpointer(self.model).load(self.cfg.PSEUDO_LABELING.BURN_IN_STUDENT_WEIGHTS)
                elif (self.iter - self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS) % self.cfg.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ == 0:
                    if self.eval_hook.distillation_burn_in:
                        self._update_teacher_model(keep_rate=self.cfg.PSEUDO_LABELING.EMA_KEEP_RATE)

            # Get pseudo-labeled data
            if self.cfg.INPUT.DATASET_MAPPER_NAME == "m2f":
                pseudo_labeled_data = self.m2f_pseudo_label()
            else:
                pseudo_labeled_data = self.mrcnn_pseudo_label()

            # Forward pass on labeled and unlabeled data
            record_dict = {}
            with autocast(dtype=self.precision):
                labeled_loss_dict = self.model(labeled_data)
                unlabeled_loss_dict = self.model(pseudo_labeled_data)
                record_dict["labeled"] = labeled_loss_dict
                record_dict["unlabeled"] = unlabeled_loss_dict
                # Process losses
                loss_dict = {}
                for key in record_dict.keys():
                    # Weighting here later
                    if key == "unlabeled":
                        #for loss_key in record_dict[key].keys():
                        #    if loss_key == "loss_mask":
                        #        record_dict[key][loss_key] = mean_met_val* record_dict[key][loss_key]
                        #loss_dict[key] = sum(record_dict[key].values())
                        #print(loss_dict[key], mean_met_val)
                        loss_dict[key] = sum(record_dict[key].values())
                    else:
                        loss_dict[key] = sum(record_dict[key].values())

                losses = sum(loss_dict.values())
       
        if self.cfg.PSEUDO_LABELING.METRIC_USE == "dynamic":
            if self.iter % self.cfg.TEST.EVAL_PERIOD == 0:
                if self.iter != 0:
                    self.metric_thresh = self.metric_mean_val - self.cfg.PSEUDO_LABELING.METRIC_OFFSET
                    print("### NEW_THRESH ######################################")
                    print(self.metric_thresh)
                    self.metric_mean_count = 0
                    self.metric_mean_acc = 0

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

def visualize_and_save_instances(unlabeled_data, save_dir="visualized_outputs"):
    """
    Visualize images with generated bit masks overlaid and save them.
    Args:
        unlabeled_data (list): List of unlabeled data containing images and instances.
        save_dir (str): Directory to save the visualized images.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, data in enumerate(unlabeled_data):
        # Extract the image and instances
        image_tensor = data["image"]  # Shape (C, H, W)
        instances = data.get("instances", None)

        if instances is None:
            print(f"No instances found for image {idx}. Skipping visualization.")
            continue

        # Check if the instances have masks
        if not hasattr(instances, "gt_masks"):
            print(f"No masks found for image {idx}. Skipping visualization.")
            continue
        
        # Extract the bitmasks
        bitmasks = instances.gt_masks  # BitMasks object

        # Convert image tensor to numpy array and transpose from (C, H, W) to (H, W, C)
        image = image_tensor.cpu().numpy().transpose(1, 2, 0)

        # Normalize the image for display
        image = (image - image.min()) / (image.max() - image.min())  # Normalize between 0 and 1
        image = (image * 255).astype(np.uint8)

        # Create an empty mask overlay with the same shape as the image
        mask_overlay = np.zeros_like(image)

        # Loop through each bitmask and overlay it on the image
        for mask_idx in range(bitmasks.tensor.shape[0]):
            mask = bitmasks.tensor[mask_idx].cpu().numpy()  # Each mask is (H, W)

            # Generate a random color for each mask
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

            # Apply the mask color only where the mask is True
            mask_overlay[mask == 1] = color  # Overlay mask regions with random color

        # Combine the mask overlay with the original image
        combined_image = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)

        # Save the image with overlaid masks
        save_path = os.path.join(save_dir, f"visualized_image_{idx}_masks.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(combined_image)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to prevent memory issues with many images

        print(f"Saved visualized image {idx} with masks to {save_path}")

