"""
Details:
"""
# imports
import os 
import cv2
import numpy as np
# maybe matplotlib
import torch

import torch.nn.functional as F

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from baseline_dev.config import add_my_config

from pseudo_labeling.engine.trainer import PseudoTrainer
from mask2former.config import add_maskformer2_config
from pseudo_labeling.config import add_pseudo_config
from mask2former import MaskFormer

# classes
# CLASS GOES HERE

# main
def main(cfg_path, weight_path, source_dir, targ_img_dir):
    # retrieve config
    cfg = setup(cfg_path, weight_path)
    predictor = DefaultPredictor(cfg)

    end_count = 0

    for img_file in os.listdir(source_dir):
        # get prediction on valid images

        if img_file.endswith(".json"):
            continue
        img = cv2.imread(os.path.join(source_dir, img_file))
        output = predictor(img)
        
        # save base image
        base_image_out_route = os.path.join(targ_img_dir, img_file)
        #cv2.imwrite(base_image_out_route, img)

        preds = output["instances"].to("cpu")

        pred_scores = preds.scores.detach().numpy()
        pred_masks = preds.pred_masks.detach().numpy()
        pred_boxes = preds.pred_boxes.tensor.detach().numpy()
        pred_classes = preds.pred_classes.detach().numpy()

        # Filter masks by prediction score
        cf_pred_scores = []
        cf_pred_masks = []
        cf_pred_boxes = []
        cf_pred_classes = []

        for j in range(len(pred_scores)):    
            
            if pred_scores[j] < 0.5:
                continue

            cf_pred_scores.append(pred_scores[j])
            cf_pred_masks.append(pred_masks[j])
            cf_pred_boxes.append(pred_boxes[j])
            cf_pred_classes.append(pred_classes[j])

        # Filter masks by metric
        mf_pred_scores = []
        mf_pred_binary_masks = []
        mf_pred_logit_masks = []
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

            # Get volumetric symmetry metric
            higher_volume = mask[mask >= 0.5] - 0.5
            lower_volume = np.minimum(mask, 0.5)
            vol_sym = conf_score * (higher_volume.sum() / lower_volume.sum()) ** 2

            if vol_sym > 0.7:
                continue

            mf_pred_scores.append(cf_pred_scores[j])
            mf_pred_binary_masks.append(binary_mask)
            mf_pred_logit_masks.append(mask)
            mf_pred_boxes.append(cf_pred_boxes[j])
            mf_pred_classes.append(cf_pred_classes[j])
            mf_vol_sym.append(vol_sym)

        # process mask data
        for i, mask in enumerate(mf_pred_logit_masks):

            # process mask 
            mask_arr = mask
            mask_arr = mask_arr * 255
            height, width = mask_arr.shape
            
            band_1 = np.where((mask_arr < 10) & (mask_arr >= 1)) 
            band_2 = np.where((mask_arr < 20) & (mask_arr >= 10)) 
            band_3 = np.where((mask_arr < 30) & (mask_arr >= 20)) 
            band_4 = np.where((mask_arr < 40) & (mask_arr >= 30)) 
            band_5 = np.where((mask_arr < 50) & (mask_arr >= 40)) 
            band_6 = np.where((mask_arr < 60) & (mask_arr >= 50)) 
            band_7 = np.where((mask_arr < 70) & (mask_arr >= 60)) 
            band_8 = np.where((mask_arr < 80) & (mask_arr >= 70)) 
            band_9 = np.where((mask_arr < 90) & (mask_arr >= 80)) 
            band_10 = np.where((mask_arr < 100) & (mask_arr >= 90)) 
            band_11 = np.where((mask_arr < 110) & (mask_arr >= 100))
            band_12 = np.where((mask_arr < 120) & (mask_arr >= 110))
            band_13 = np.where((mask_arr < 130) & (mask_arr >= 120))
            band_14 = np.where((mask_arr < 140) & (mask_arr >= 130))
            band_15 = np.where((mask_arr < 150) & (mask_arr >= 140))
            band_16 = np.where((mask_arr < 160) & (mask_arr >= 150))
            band_17 = np.where((mask_arr < 170) & (mask_arr >= 160))
            band_18 = np.where((mask_arr < 180) & (mask_arr >= 170))
            band_19 = np.where((mask_arr < 190) & (mask_arr >= 180))
            band_20 = np.where((mask_arr < 200) & (mask_arr >= 190))
            band_21 = np.where((mask_arr < 210) & (mask_arr >= 200)) 
            band_22 = np.where((mask_arr < 220) & (mask_arr >= 210))
            band_23 = np.where((mask_arr < 230) & (mask_arr >= 220))
            band_24 = np.where((mask_arr < 240) & (mask_arr >= 230))
            band_25 = np.where((mask_arr <= 255) & (mask_arr >= 240))

            bands = [
                band_1,
                band_2,
                band_3,
                band_4,
                band_5,
                band_6,
                band_7,
                band_8,
                band_9,
                band_10,
                band_11,
                band_12,
                band_13,
                band_14,
                band_15,
                band_16,
                band_17,
                band_18,
                band_19,
                band_20,
                band_21, 
                band_22,
                band_23,
                band_24,
                band_25
            ]

            colours = [
                [255, 0, 0],
                [212, 0, 42],
                [170, 0, 85],
                [127, 0, 127],
                [85, 0, 170],
                [42, 0, 212],
                [0, 0, 255],
                [0, 0, 255],
                [0, 21, 255],
                [0, 42, 255],
                [0, 63, 255],
                [0, 84, 255],
                [0, 105, 255],
                [0, 255, 0], 
                [0, 255, 0],
                [0, 255, 0], 
                [0, 255, 0],
                [0, 255, 0], 
                [0, 255, 0],
                [0, 255, 0], 
                [0, 255, 0],
                [0, 255, 0], 
                [0, 255, 0], 
                [0, 255, 0],
                [0, 255, 0]
            ]

            # Coloring the mask: Red for bad, Green for good, Blue for background
            # Bad - Red
            coloured_mask_img = img.copy()
            for band, colour in zip(bands, colours):
                temp_maks = np.zeros((height, width, 1), dtype=np.uint8)
                temp_maks[band[0], band[1]] = 255
                temp_maks = cv2.threshold(temp_maks, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                coloured_mask_img[temp_maks==255] = colour
            
            coloured_mask_img = cv2.addWeighted(img, 0.7 ,coloured_mask_img, 0.3, 0)

                #coloured_mask_img[band[0], band[1], 0] = colour[0]
                #coloured_mask_img[band[0], band[1], 1] = colour[1]
                #coloured_mask_img[band[0], band[1], 2] = colour[2]

            # Set mask title and save path
            mask_title = f"{img_file}_coloured_mask_{i}.png"
            mask_out_route = os.path.join(targ_img_dir, mask_title)

            # Save the colored mask image
            cv2.imwrite(mask_out_route, coloured_mask_img)
                
        if end_count > 1:
            break
        end_count += 1


def setup(config_path, weights_path):
    """ Initialise config and ammend based on command line arguments """
    cfg = get_cfg()
    # adding argments to base config to accomodate pseudo labeling
    add_maskformer2_config(cfg)
    add_pseudo_config(cfg)
    cfg.merge_from_file(config_path)  
    cfg.MODEL.WEIGHTS = weights_path
    return cfg
     
# execute
if __name__ == "__main__":
    main(
        "configs/pseudo_labeling/config_files/ps_m2f.yaml",
        "outputs/m2f_test/baseline/pre_training_best_model.pth",
        "datasets/jr_v5_unlabeled_data",
        "results_store/applying_metric_to_m2f"
    )
