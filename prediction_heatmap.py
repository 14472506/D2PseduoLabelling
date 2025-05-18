import os
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

import torch.nn.functional as F

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from pseudo_labeling.config import add_pseudo_config
from pseudo_labeling.modelling.my_rcnn import MyGeneralizedRCNN
from pseudo_labeling.modelling import mask_head
from pseudo_labeling.modelling import custom_roi


def main(cfg_path, weight_path, source_dir, targ_img_dir):
    """
    Processes images using a Detectron2 instance segmentation model.
    For each prediction that passes filtering, the script produces and saves:
      1. A heat map overlay using a colorblind-friendly viridis colormap.
      2. An image with the binary mask (contours) and the confidence score overlayed.
      3. A 3D surface plot of the logit mask for the localized nonzero region.
      4. A 2D cross-section (from the middle row of the localized area) of the logit mask.
    """
    cfg = setup(cfg_path, weight_path)
    predictor = DefaultPredictor(cfg)
    predictor.model.pseudo_labeling = True

    end_count = 0

    for img_file in os.listdir(source_dir):
        if img_file.endswith(".json"):
            continue

        img_path = os.path.join(source_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue  # skip invalid images

        output = predictor(img)
        preds = output["instances"].to("cpu")

        pred_scores  = preds.scores.detach().numpy()
        pred_masks   = preds.pred_masks.detach().numpy() / 255  # normalized logits (0-1)
        pred_boxes   = preds.pred_boxes.tensor.detach().numpy()
        pred_classes = preds.pred_classes.detach().numpy()

        # Filter predictions by score (you can re-enable thresholding if needed)
        cf_pred_scores   = []
        cf_pred_masks    = []
        cf_pred_boxes    = []
        cf_pred_classes  = []
        for j in range(len(pred_scores)):
            # Uncomment the following line if you wish to filter by score:
            # if pred_scores[j] < 0.5: continue
            cf_pred_scores.append(pred_scores[j])
            cf_pred_masks.append(pred_masks[j])
            cf_pred_boxes.append(pred_boxes[j])
            cf_pred_classes.append(pred_classes[j])

        # Further filter using volumetric symmetry metric
        mf_pred_scores       = []
        mf_pred_binary_masks = []
        mf_pred_logit_masks  = []
        mf_pred_boxes        = []
        mf_pred_classes      = []
        mf_vol_sym           = []

        for j in range(len(cf_pred_scores)):
            conf_score = cf_pred_scores[j]
            mask = cf_pred_masks[j]

            # Create a binary mask for thresholded regions
            binary_mask = np.where(mask >= 0.5, 1, 0)
            area = np.count_nonzero(binary_mask)
            if area < 50:
                continue

            good_volume = mask[mask >= 0.5]
            bad_volume  = mask
            vol_sym = conf_score * (good_volume.sum() / bad_volume.sum()) ** 3

            if vol_sym < 0.50:
                continue

            mf_pred_scores.append(conf_score)
            mf_pred_binary_masks.append(binary_mask)
            mf_pred_logit_masks.append(mask)
            mf_pred_boxes.append(cf_pred_boxes[j])
            mf_pred_classes.append(cf_pred_classes[j])
            mf_vol_sym.append(vol_sym)

        # Process each filtered instance
        for idx, (logit_mask, binary_mask, score, pred_class) in enumerate(
            zip(mf_pred_logit_masks, mf_pred_binary_masks, mf_pred_scores, mf_pred_classes)
        ):
            # ---------------------------
            # 1. Heat map overlay using viridis
            # ---------------------------
            mask_arr = (logit_mask * 255).astype(np.uint8)
            height, width = mask_arr.shape

            # Create intensity bands (25 bands)
            bands = [
                np.where((mask_arr < 10)  & (mask_arr >= 1)),
                np.where((mask_arr < 20)  & (mask_arr >= 10)),
                np.where((mask_arr < 30)  & (mask_arr >= 20)),
                np.where((mask_arr < 40)  & (mask_arr >= 30)),
                np.where((mask_arr < 50)  & (mask_arr >= 40)),
                np.where((mask_arr < 60)  & (mask_arr >= 50)),
                np.where((mask_arr < 70)  & (mask_arr >= 60)),
                np.where((mask_arr < 80)  & (mask_arr >= 70)),
                np.where((mask_arr < 90)  & (mask_arr >= 80)),
                np.where((mask_arr < 100) & (mask_arr >= 90)),
                np.where((mask_arr < 110) & (mask_arr >= 100)),
                np.where((mask_arr < 120) & (mask_arr >= 110)),
                np.where((mask_arr < 130) & (mask_arr >= 120)),
                np.where((mask_arr < 140) & (mask_arr >= 130)),
                np.where((mask_arr < 150) & (mask_arr >= 140)),
                np.where((mask_arr < 160) & (mask_arr >= 150)),
                np.where((mask_arr < 170) & (mask_arr >= 160)),
                np.where((mask_arr < 180) & (mask_arr >= 170)),
                np.where((mask_arr < 190) & (mask_arr >= 180)),
                np.where((mask_arr < 200) & (mask_arr >= 190)),
                np.where((mask_arr < 210) & (mask_arr >= 200)),
                np.where((mask_arr < 220) & (mask_arr >= 210)),
                np.where((mask_arr < 230) & (mask_arr >= 220)),
                np.where((mask_arr < 240) & (mask_arr >= 230)),
                np.where((mask_arr <= 255) & (mask_arr >= 240))
            ]

            # Generate 25 colorblind-friendly colors from viridis (convert to BGR for OpenCV)
            n_colors = 25
            viridis = plt.get_cmap('viridis', n_colors)
            colours = []
            for i in range(n_colors):
                r, g, b, _ = viridis(i)
                colours.append([int(b * 255), int(g * 255), int(r * 255)])

            # Create the heat map image by overlaying color bands
            coloured_mask_img = img.copy()
            for band, colour in zip(bands, colours):
                temp_mask = np.zeros((height, width), dtype=np.uint8)
                temp_mask[band[0], band[1]] = 255
                temp_mask = cv2.threshold(temp_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                coloured_mask_img[temp_mask == 255] = colour
            coloured_mask_img = cv2.addWeighted(img, 0.7, coloured_mask_img, 0.3, 0)
            heatmap_filename = os.path.join(targ_img_dir, f"{img_file}_heatmap_{idx}.png")
            cv2.imwrite(heatmap_filename, coloured_mask_img)

            # ---------------------------
            # 2. Binary mask overlay with confidence score
            # ---------------------------
            binary_overlay = img.copy()
            # Create binary image for contour detection (0 or 255)
            bin_mask = (binary_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contours (in red) on the image
            cv2.drawContours(binary_overlay, contours, -1, (0, 0, 255), 2)
            # Determine text position using the largest contour (if available)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                text_position = (x, y - 10 if y - 10 > 10 else y + 10)
            else:
                text_position = (10, 30)
            overlay_text = f"Score: {score:.2f}"
            cv2.putText(binary_overlay, overlay_text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            binary_overlay_filename = os.path.join(targ_img_dir, f"{img_file}_binary_overlay_{idx}.png")
            cv2.imwrite(binary_overlay_filename, binary_overlay)

            # ---------------------------
            # 3. 3D surface plot of the localized logit mask
            # ---------------------------
            # Localize to the area with nonzero values
            nonzero = np.nonzero(mask_arr)
            if nonzero[0].size > 0:
                y_min, y_max = np.min(nonzero[0]), np.max(nonzero[0])
                x_min, x_max = np.min(nonzero[1]), np.max(nonzero[1])
            else:
                y_min, y_max = 0, height - 1
                x_min, x_max = 0, width - 1

            # Extract the localized region from the logit mask
            mask_local = mask_arr[y_min:y_max+1, x_min:x_max+1]
            local_height, local_width = mask_local.shape

            # Create a meshgrid corresponding to the localized region
            X_local, Y_local = np.meshgrid(np.arange(x_min, x_max+1), np.arange(y_min, y_max+1))
            Z_local = mask_local

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X_local, Y_local, Z_local, cmap='viridis', edgecolor='none')
            # Remove axis markings and titles
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_title('')
            plot3d_filename = os.path.join(targ_img_dir, f"{img_file}_3d_logit_local_{idx}.png")
            plt.savefig(plot3d_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # ---------------------------
            # 4. 2D cross-section plot (middle row of the localized region)
            # ---------------------------
            mid_local = local_height // 2
            cross_section = mask_local[mid_local, :]
            fig2 = plt.figure()
            # Plot the cross-section against the localized x-range
            plt.plot(np.arange(x_min, x_max+1), cross_section, color='black')
            plt.xticks([])  # remove x-axis ticks
            plt.yticks([])  # remove y-axis ticks
            plt.title('')
            cross_section_filename = os.path.join(targ_img_dir, f"{img_file}_logit_cross_section_local_{idx}.png")
            plt.savefig(cross_section_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig2)

        print(f"Processed image count: {end_count}")
        end_count += 1
        if end_count > 500:
            break


def setup(cfg_path, weights_path):
    cfg = get_cfg()
    add_pseudo_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    return cfg


if __name__ == "__main__":
    main(
        "configs/pseudo_labeling/config_files/ps_mrcnn.yaml",
        "outputs/mrcnn_gen/pt_mrcnn/burn_in_best_model.pth",
        "datasets/jr_ds_v6/images",
        "results_store/for_paper"
    )

