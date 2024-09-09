# imports
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from pseudo_labeling.config import add_pseudo_config
from pseudo_labeling.modelling.my_rcnn import MyGeneralizedRCNN
from pseudo_labeling.modelling import mask_head 
from pseudo_labeling.modelling import custom_roi

# main
def main(cfg_path, weight_path, source_dir, targ_img_dir):
    # retrieve config
    cfg = setup(cfg_path, weight_path)
    predictor = DefaultPredictor(cfg)

    predictor.model.pseudo_labeling = True
    vol_sym_values = []

    count = 0 

    for img_file in os.listdir(source_dir):
        # get prediction on valid images

        if img_file.endswith(".json"):
            continue
        img = cv2.imread(os.path.join(source_dir, img_file))
        output = predictor(img)
        
        preds = output["instances"].to("cpu")

        pred_scores = preds.scores.detach().numpy()
        pred_masks = preds.pred_masks.detach().numpy() / 255

        # Filter masks by prediction score
        cf_pred_scores = []
        cf_pred_masks = []

        for j in range(len(pred_scores)):    
            
            if pred_scores[j] < 0.5:
                continue

            cf_pred_scores.append(pred_scores[j])
            cf_pred_masks.append(pred_masks[j])

        # Filter masks by metric
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
            good_volume = mask[mask >= 0.5]
            bad_volume = mask
            vol_sym = conf_score * (good_volume.sum()/bad_volume.sum())**3            

            # Add vol_sym to the list of all metric values
            vol_sym_values.append(vol_sym)

        count += 1
        print(count)
    
    # Plot the distribution of the vol_sym metric
    plt.figure(figsize=(10, 6))
    plt.hist(vol_sym_values, bins=100, color='blue', alpha=0.7)
    plt.title('Distribution of Volumetric Symmetry Metric Across All Images')
    plt.xlabel('Volumetric Symmetry Metric (vol_sym)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save the plot as an image
    plt.savefig("050_burn_in.png")
    plt.close()

def setup(cfg_path, weights_path):
    cfg = get_cfg()
    add_pseudo_config(cfg)
    cfg.merge_from_file(cfg_path)    
    cfg.MODEL.WEIGHTS = weights_path 
    return(cfg)
     
# execute
if __name__ == "__main__":
    main(
        "configs/pseudo_labeling/config_files/ps_mrcnn.yaml",
        "outputs/mrcnn_ps_exps/all_stages_1/burn_in_best_model.pth",
        "datasets/jr_v5_unlabeled_data",
        ""
    )