"""
Details
"""
# imports 
import logging
import operator

import torch

from detectron2.data.build import get_detection_dataset_dicts, build_batch_data_loader, worker_init_reset_seed
from detectron2.utils.logger import _log_api_usage
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import TrainingSampler
from detectron2.data.common import DatasetFromList, MapDataset, ToIterableDataset
from detectron2.config import configurable

from pseudo_labeling.data.unlabelled_mapper import UnlabeledDatasetMapper
from pseudo_labeling.data.common import AspectRatioGroupedSemiSupDatasetTwoCrop

# functions
# loader and pseudo loader from config
def _pseudo_train_loader_from_config(cfg, 
                                     labeled_mapper=None, 
                                     unlabeled_mapper=None, 
                                     *, 
                                     labeled_dataset=None, 
                                     unlabeled_dataset=None, 
                                     labeled_sampler=None,
                                     unlabeled_sampler=None):
    """
    Details 
    """
    # Handle labeled dataset 
    if labeled_dataset is None:

        labeled_dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FIconfigurableES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    # Handle unlabeled dataset 
    if unlabeled_dataset is None:

        unlabeled_dataset = get_detection_dataset_dicts(
            cfg.PSEUDO_LABELING.DATASET,
            filter_empty = False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FIconfigurableES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.PSEUDO_LABELING.DATASET[0])

    # Handle labeled mapper
    if labeled_mapper is None:
        labeled_mapper = DatasetMapper(cfg, True)
    if unlabeled_mapper is None:
        unlabeled_mapper = UnlabeledDatasetMapper(cfg)
    
    # Handle Labeled Sampler
    if labeled_sampler is None:
        
        labeled_sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)

        if isinstance(labeled_dataset, torch.utils.data.IterableDataset):
            logger.info("Not using any sampler since the labeled dataset is IterableDataset.")
            labeled_sampler = None
        else:
            logger.info("Using training sampler {} for labeled".format(labeled_sampler_name))
            labeled_sampler = TrainingSampler(len(labeled_dataset))

    # Handle Unlabeled Sampler
    if unlabeled_sampler is None:
        
        unlabeled_sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)

        if isinstance(unlabeled_dataset, torch.utils.data.IterableDataset):
            logger.info("Not using any sampler since the unlabeled dataset is IterableDataset.")
            unlabeled_sampler = None
        else:
            logger.info("Using training sampler {} for unlabeled".format(unlabeled_sampler_name))
            unlabeled_sampler = TrainingSampler(len(unlabeled_dataset))
    
    return {
        "labeled_dataset": labeled_dataset,
        "unlabeled_dataset": unlabeled_dataset,
        "labeled_mapper": labeled_mapper,
        "unlabeled_mapper": unlabeled_mapper,
        "labeled_sampler": labeled_sampler,
        "unlabeled_sampler": unlabeled_sampler,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }

@configurable(from_config=_pseudo_train_loader_from_config)
def build_pseudo_train_loader(
    labeled_dataset,
    unlabeled_dataset,
    *,
    labeled_mapper,
    unlabeled_mapper,
    labeled_sampler=None,
    unlabeled_sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    **kwargs
):
    """
    Detials
    """
    # Handle labeled elements
    if isinstance(labeled_dataset, list):
        labeled_dataset = DatasetFromList(labeled_dataset, copy=False)
        print("labeled_ds_from_list")
    if labeled_mapper is not None:
        labeled_dataset = MapDataset(labeled_dataset, labeled_mapper)
        print("labeled_mapper")

    if isinstance(labeled_dataset, torch.utils.data.IterableDataset):
        assert labeled_sampler is None, "sampler must be None if dataset is IterableDataset"
        print("labeled_iterable_ds")
    else:
        if labeled_sampler is None:
            labeled_sampler = TrainingSampler(len(labeled_dataset))
            print("labeled_sampler")
        assert isinstance(labeled_sampler, torch.utils.data.Sampler), f"Expect a Sampler but got {type(labeled_sampler)}"

    # Handle unlabeled elements
    if isinstance(unlabeled_dataset, list):
        unlabeled_dataset = DatasetFromList(unlabeled_dataset, copy=False)
        print("unlabeled_ds_from_list")
    if unlabeled_mapper is not None:
        unlabeled_dataset = MapDataset(unlabeled_dataset, unlabeled_mapper)
        print("unlabeled_mapper")

    if isinstance(unlabeled_dataset, torch.utils.data.IterableDataset):
        assert unlabeled_sampler is None, "sampler must be None if dataset is IterableDataset"
        print("unlabeled_iterable_ds")
    else:
        if unlabeled_sampler is None:
            unlabeled_sampler = TrainingSampler(len(unlabeled_dataset))
            print("unlabeled_sampler")
        assert isinstance(unlabeled_sampler, torch.utils.data.Sampler), f"Expect a Sampler but got {type(unlabeled_sampler)}"
    
    #if aspect_ratio_grouping:
    #labeled_batch_data_loader = torch.utils.data.DataLoader(
    #    labeled_dataset,
    #    sampler=labeled_sampler,
    #    num_workers=num_workers,
    #    batch_sampler=None,
    #    collate_fn=operator.itemgetter(0),
    #    worker_init_fn=worker_init_reset_seed,
    #)

    labeled_batch_data_loader = build_batch_data_loader(
        labeled_dataset,
        labeled_sampler,
        total_batch_size,
        aspect_ratio_grouping = False,
        num_workers = num_workers,
    )

    unlabeled_batch_data_loader = build_batch_data_loader(
        unlabeled_dataset,
        unlabeled_sampler,
        total_batch_size,
        aspect_ratio_grouping = False,
        num_workers = num_workers,
    )

    #unlabeled_batch_data_loader = torch.utils.data.DataLoader(
    #    unlabeled_dataset,
    #    sampler=unlabeled_sampler,
    #    num_workers=num_workers,
    #    batch_sampler=None,
    #    collate_fn=operator.itemgetter(0),
    #    worker_init_fn=worker_init_reset_seed,
    #)
    return (labeled_batch_data_loader, unlabeled_batch_data_loader)

