"""
Detials
"""
# imports
import os 
import cv2
import numpy as np
import json
import math
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import pycocotools.mask as mask_utils

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import pairwise_iou, Boxes, PolygonMasks, BitMasks

from baseline_dev.config import add_my_config

# needed for registration
from baseline_dev.modeling.my_rcnn import MyGeneralizedRCNN

# classes
class PredictionAnyliser():
    """ Detials """
    def __init__(self, predictor, src_route, out_route):
        """ Detials """
        self.predictor = predictor
        self.src_route = src_route
        self.out_route = out_route

    # =========================================================================
    # Main Execution
    # =========================================================================
    def _image_loop(self):
        """ Detials """
        # task parameter initialisation
        if self.task == "task":
            self._task_init()

        for img_file in os.listdir(self.src_route):
            # skip non image files
            if img_file.endswith(".json"):
                continue
            # get image
            img = cv2.imread(os.path.join(self.src_route, img_file))
            output = self.predictor(img)
            # do task
            if self.task == "task":
                try:
                    self._task(img_file, img, output)
                except KeyError:
                    continue

        if self.task == "task":

            print("plotting")
            print("sactter")
            coef = np.polyfit(self.results["mask_ious"], self.results["metric_vals"], 1)
            polynomial = np.poly1d(coef)
            x_lin_reg = np.array(self.results["mask_ious"])
            y_lin_reg = polynomial(x_lin_reg)
            plt.scatter(self.results["mask_ious"], self.results["metric_vals"])
            plt.plot(x_lin_reg, y_lin_reg, color="red")  # linear regression line in red
            plt.title('Mask_IoU and Metric Scatter plot')
            plt.xlabel('Mask IoU')
            plt.ylabel('Metric')
            
            # Save the figure
            plt.savefig('scatter_dev.png', dpi=300) # Save as PNG with high resolution
            plt.close() # Close the figure window to

            print("line")
            zipped = zip(self.results["mask_ious"], self.results["metric_vals"])
            zipped_list = sorted(zipped, key=lambda x: x[0])
            ious, metrics = zip(*zipped_list)

            count = list(range(0, len(ious)))

            fig, ax1 = plt.subplots(figsize=(10, 6))  # Creates a figure and a set of subplots

            color = 'tab:red'
            ax1.set_xlabel('Number of Instances')
            ax1.set_ylabel('IoU', color=color)
            ax1.plot(count, ious, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Metric', color=color)  # We already handled the x-label with ax1
            ax2.plot(count, metrics, color=color, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # Optional: Adjust the padding between and around subplots
            plt.title('IoU and Paired Metric Plot')
            plt.legend()
            
            # Save the figure
            plt.savefig('line_dev.png', dpi=300) # Save as PNG with high resolution
            plt.close() # Close the figure window to

            good_accepted = 0
            bad_accepts = 0
            good_rejected = 0
            bad_rejects = 0
            rejected = 0
            good_id = 0
            bad_id = 0

            met_thresh = 0.5
            for i, iou in enumerate(ious):
                if iou >= 0.8 and metrics[i] >= met_thresh:
                    good_id += 1
                    good_accepted += 1
                elif iou < 0.8 and metrics[i] < met_thresh:
                    good_id += 1
                    good_rejected += 1
                else:
                    bad_id += 1
                    if iou >= 0.8 and metrics[i] <= met_thresh:
                        bad_rejects += 1
                    if iou < 0.8 and metrics[i] > met_thresh:
                        bad_accepts += 1
            
            total = good_id + bad_id
            good_perc = (100/total) * good_id
            bad_perc = 100 - good_perc

            print(f"{good_perc}% good ids and {bad_perc}% bad ids")
            print(f"{good_accepted} good accepted")
            print(f"{good_rejected} good rejected")
            print(f"{bad_accepts} bad accepted")
            print(f"{bad_rejects} bad rejected")

    # =========================================================================
    # TASK
    # =========================================================================
    def do_task(self):
        """ Detials """
        self.task = "task"
        self._image_loop()

    def _task_init(self):
        """ Detials """
        for file_path in os.listdir(self.src_route):
            if file_path.endswith(".json"):
                with open(os.path.join(self.src_route, file_path), "r") as file:
                    self.gt_dict = json.load(file)

        data_index = {}
        for img_data in self.gt_dict["images"]:
            matching_ids = []
            for idx, annotation in enumerate(self.gt_dict["annotations"]):
                if img_data["id"] == annotation["image_id"]:
                    matching_ids.append(annotation["id"])
            data_index[img_data["file_name"]] = matching_ids
        self.data_index = data_index

        self.results = {
            "mask_ious": [],
            "metric_vals": []
        }

    def _task(self, file_path, image, output):
        """ Detials """
        # prediction pre processing
        preds = output["instances"].to("cpu")
        pred_scores = preds.scores.numpy()
        pred_masks = preds.pred_masks.numpy()/255
        pred_boxes = preds.pred_boxes
        _, height, width = pred_masks.shape
        logit_masks = pred_masks
        pred_masks = [(mask >= 0.5).astype(np.uint8) for mask in pred_masks]

        # pre processing ground truth data
        raw_masks, raw_boxes = self._get_raw_image_anns(file_path)
        raw_boxes = [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in raw_boxes]
        gt_boxes = Boxes(raw_boxes)
        gt_masks = [self._mask_to_bitmask(mask, height, width) for mask in raw_masks]
        gt_masks = [gt_mask[:,:,0] for gt_mask in gt_masks]

        # getting bounding boxes iou based matching indexes
        gt_index = self._box_iou_index(gt_boxes, pred_boxes)

        # getting mask ious
        self._get_results(gt_masks, pred_masks, pred_scores, logit_masks, gt_index)
        print(f"{file_path} completed")
            
    def _get_raw_image_anns(self, file_path):
        """Detials """        
        # get instance masks
        get_mask = []
        get_boxes = []
        for ann_id in self.data_index[file_path]:
            get_mask.append(self.gt_dict["annotations"][ann_id]["segmentation"])
            get_boxes.append(self.gt_dict["annotations"][ann_id]["bbox"])
        return get_mask, get_boxes
    
    def _mask_to_bitmask(self, mask, height, width):
        """ Detials """
        # polygone or rle format
        if isinstance(mask, list):
            rle = mask_utils.frPyObjects(mask, height, width)
            mask = mask_utils.decode(rle)
        elif isinstance(mask, dict):
            mask = mask_utils.decode(rle)
        return mask
    
    def _box_iou_index(self, gt_boxes, pred_boxes):
        """ Detials """
        index_dict = {}
        for i ,gt_box in enumerate(gt_boxes):
            for j, p_box in enumerate(pred_boxes):
                
                xA = max(p_box[0], gt_box[0])
                yA = max(p_box[1], gt_box[1])
                xB = min(p_box[2], gt_box[2])
                yB = min(p_box[3], gt_box[3])

                int_area = max(0, xB-xA+1) * max(0, yB-yA+1)
                if int_area == 0:
                    continue

                boxA_area = (p_box[2]-p_box[0]+1) * (p_box[3]-p_box[1]+1) 
                boxB_area = (gt_box[2]-gt_box[0]+1) * (gt_box[3]-gt_box[1]+1)

                union_area = boxA_area + boxB_area - int_area

                iou = int_area/union_area
                if iou >= 0.15:
                    if i in index_dict:
                        index_dict[i].append(j)
                    else:
                        index_dict[i] = [j]
        return index_dict
    
    def _get_results(self, gt_masks, pred_masks, pred_scores, logit_masks, gt_index):
        """ Detaisl """
        ious = []
        metrics = []
        for gt_idx, pred_idxs in gt_index.items():
            gt_mask = gt_masks[gt_idx]
            for pred_idx in pred_idxs:
                # getting ious
                pred_mask = pred_masks[pred_idx]
                inter = np.logical_and(pred_mask, gt_mask).sum()
                if inter == 0:
                    continue
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = inter / union if union > 0 else 0
                if iou > 0.5:
                    ious.append(iou)
                else:
                    # dont do anymore if the iou is not recorded
                    continue

                # getting metric results 
                #metric = math.sqrt(pred_scores[pred_idx] * np.mean(logit_masks[pred_idx][pred_mask.astype(bool)]))

                ########################################################################################

                #base_mask = logit_masks[pred_idx] > 0
                #inter_met = np.logical_and(pred_mask, base_mask).sum()
                #union_met = np.logical_or(pred_mask, base_mask).sum()
                #metric = inter_met / union_met

                #########################################################################################

                base_band = logit_masks[pred_idx] > 0
                band_01 = (logit_masks[pred_idx] > 0) & (logit_masks[pred_idx] <= 0.10)
                band_02 = (logit_masks[pred_idx] > 0.10) & (logit_masks[pred_idx] <= 0.20)
                band_03 = (logit_masks[pred_idx] > 0.20) & (logit_masks[pred_idx] <= 0.30)
                band_04 = (logit_masks[pred_idx] > 0.30) & (logit_masks[pred_idx] <= 0.40)
                band_05 = (logit_masks[pred_idx] > 0.40) & (logit_masks[pred_idx] <= 0.50)
                band_06 = (logit_masks[pred_idx] > 0.50) & (logit_masks[pred_idx] <= 0.60)
                band_07 = (logit_masks[pred_idx] > 0.60) & (logit_masks[pred_idx] <= 0.70)
                band_08 = (logit_masks[pred_idx] > 0.70) & (logit_masks[pred_idx] <= 0.80)
                band_09 = (logit_masks[pred_idx] > 0.80) & (logit_masks[pred_idx] <= 0.90)
                band_10 = logit_masks[pred_idx] > 0.9

                base_val = logit_masks[pred_idx][base_band].sum()
                val_01 = logit_masks[pred_idx][band_01].sum() * 0.01
                val_02 = logit_masks[pred_idx][band_02].sum() * 0.04
                val_03 = logit_masks[pred_idx][band_03].sum() * 0.09
                val_04 = logit_masks[pred_idx][band_04].sum() * 0.16
                val_05 = logit_masks[pred_idx][band_05].sum() * 0.25
                val_06 = logit_masks[pred_idx][band_06].sum() * 0.36
                val_07 = logit_masks[pred_idx][band_07].sum() * 0.49
                val_08 = logit_masks[pred_idx][band_08].sum() * 0.64
                val_09 = logit_masks[pred_idx][band_09].sum() * 0.81
                val_10 = logit_masks[pred_idx][band_10].sum()

                metric = pred_scores[pred_idx] * (val_01+val_02+val_03+val_04+val_05+val_06+val_07+val_08+val_09+val_10)/base_val

                ###################################################################################

                #logit_mask = logit_masks[pred_idx]
                #x_grad = cv2.Sobel(logit_mask, cv2.CV_64F, 1, 0, ksize=3)
                #y_grad = cv2.Sobel(logit_mask, cv2.CV_64F, 0, 1, ksize=3)                
                #magnitured = np.sqrt(x_grad**2 + y_grad**2)
                #max_alt = np.max(logit_mask)
                #roi_mask = (logit_mask > 0) & (logit_mask <= 0.5)
                #roi_magnitude = magnitured[roi_mask]
                #print(roi_magnitude)
                #metric = (1/max_alt) * np.mean(roi_magnitude)

                #####################################################################################
                # Symmetry metric

                #logit_mask = logit_masks[pred_idx]
                #good_volume = logit_mask[logit_mask >= 0.5] - 0.5
                #bad_volume = np.minimum(logit_mask, 0.5)
                
                #metric =  pred_scores[pred_idx] * (math.sqrt((inter_met/union_met)**2) * (good_volume.sum()/bad_volume.sum()))
                #metric = pred_scores[pred_idx] * ((1.80*good_volume.sum())/(0.2*bad_volume.sum()))**2
                #metric = pred_scores[pred_idx] * (good_volume.sum()/bad_volume.sum())**2
                
                
                ######################################################################################
                # Petras

                #logit_mask = logit_masks[pred_idx]
                #good_volume = logit_mask[logit_mask >= 0.5]
                #bad_volume = logit_mask
                #metric = pred_scores[pred_idx] * (good_volume.sum()/bad_volume.sum())**2

                ######################################################################################

                print("########################")
                print(iou)
                print(metric)

                metrics.append(metric)

        self.results["mask_ious"].extend(ious)
        self.results["metric_vals"].extend(metrics)

# functions 
def main(cfg_path, weights_path, source_dir, targ_img_dir):
    """ Detials """
    # setup
    cfg = setup(cfg_path, weights_path)
    predictor = DefaultPredictor(cfg)
    pred_anyl = PredictionAnyliser(predictor, source_dir, targ_img_dir)
    pred_anyl.do_task()

    # do stuff

def setup(cfg_path, weights_path):
    """ Detials """
    cfg = get_cfg()
    add_my_config(cfg)
    cfg.merge_from_file(cfg_path)    
    cfg.MODEL.WEIGHTS = weights_path 
    return(cfg)

# execution
if __name__ == "__main__":
    main( 
        "configs/test_2.yaml",
        "outputs/DEV_TEST_3/best_model.pth",
        "datasets/jersey_dataset_v4/test",
        "results_store/pred_anaylsis_dev_bin"
        )