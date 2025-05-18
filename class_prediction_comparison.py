# imports
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.ticker as mtick  # For nice % formatting if needed

from mask2former.config import add_maskformer2_config
from pseudo_labeling.config import add_pseudo_config


def main(cfg_path, weight_path, source_dir, source_json, targ_out_dir):
    # retrieve configuration and create predictor
    cfg = setup(cfg_path, weight_path)
    predictor = DefaultPredictor(cfg)
    predictor.model.pseudo_labeling = True

    pred_score_acc = []
    count = 0

    idx_list = []
    with open(source_json, "r") as file:
        js_dict = json.load(file)
    
    for img_loc in js_dict["images"]:
        idx_list.append(img_loc["file_name"])
        
    for img_file in idx_list:
        if img_file.endswith(".json"):
            continue

        img = cv2.imread(os.path.join(source_dir, img_file))
        output = predictor(img)
        preds = output["instances"].to("cpu")
        pred_scores = preds.scores.detach().numpy()

        for score in pred_scores:
            # if score > 0.01:
            pred_score_acc.append(score)
    
        count += 1
        print("Processed image:", count)


    # Define figure size for small paper-friendly output
    width_cm = 4.5
    height_cm = 4.5
    dpi = 600
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)

    # Calculate relative frequencies
    weights = np.ones_like(pred_score_acc) / len(pred_score_acc)
    ax.hist(pred_score_acc, bins=10, weights=weights, color='tab:blue', alpha=0.8, edgecolor='black', linewidth=0.3)

    # Clean minimalist formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)  # <-- You can lock it to 1 or set max dynamically if needed

    # Set ticks to 0, 0.5, 1
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

    ax.tick_params(axis='both', which='major', labelsize=6, length=2, width=0.4, pad=1)

    # Optional: remove labels completely if you're captioning externally in LaTeX
    ax.set_xlabel('')
    ax.set_ylabel('', fontsize=7)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: set y-ticks manually for consistent spacing across datasets
    #yticks = np.round(np.linspace(0, max(weights) * 1.2, 3), 3)
    #ax.set_yticks(yticks)

    # Save high-res figure
    plt.tight_layout()
    plt.savefig("class_scores_single_class.pdf", bbox_inches="tight", dpi=dpi)
    plt.close()

def setup(cfg_path, weights_path):
    """Initialise config and amend based on provided file paths."""
    cfg = get_cfg()
    # add pseudo labeling configurations to the base config
    add_maskformer2_config(cfg)
    add_pseudo_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path 
    return cfg

     
# execute
if __name__ == "__main__":

    DATASETS = {
        "cityscapes": {
            "cfg": "configs/pseudo_labeling/config_files/ps_m2f_from_gd.yaml",
            "weights": "outputs/m2f/cityscapes_test/baseline/pre_training_best_model.pth",
            "img_dir": "datasets/cityscapes/gtFine/val",
            "json": "datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
            },
        "jr": {
            "cfg": "configs/pseudo_labeling/config_files/ps_m2f_from_gd.yaml",
            "weights": "outputs/m2f/baseline/02/pre_training_best_model.pth",
            "img_dir": "datasets/jr_ds_v6/images",
            "json": "datasets/jr_ds_v6/annotations/v6.1_val.json"
            }
        }

    # Run main on selected dataset
    selected_dataset = "jr"  # change this line only
    params = DATASETS[selected_dataset]
    main(params["cfg"], params["weights"], params["img_dir"], params["json"], "")


