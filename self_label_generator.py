"""
Datials
"""
# =========================================================
# imports
# =========================================================
# base imports
import os

# Detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import PolygonMasks

# other third party
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json

# local
from pseudo_labeling.config import add_pseudo_config
from mask2former.config import add_maskformer2_config
from mask2former import MaskFormer


# =========================================================
# class
# =========================================================
class LabelGenerator():
    """ Detials """
    # -------------------------------------------
    # initialisation and supporting
    # -------------------------------------------
    def __init__(self, cfg):
        """ Details """
        # init from config
        self._init_from_config(cfg)
        self.data_dict = self._init_data_dict()

        # something
        self.predictor = DefaultPredictor(self.model_cfg)
        self.processor = self._process_selector()

    def _init_from_config(self, cfg):
        """ Details """
        self.img_dir = cfg["img_dir"]
        self.out_dir = cfg["out_dir"]
        self.model_cfg = setup(cfg["cfg_pth"], cfg["weights_pth"])
        self.interactive = cfg["interactive"]  
        self.class_confidence_thresh = cfg["class_confidence_thresh"]

    def _process_selector(self):
        """ Details """
        if self.interactive:
            return self._interactive_processor
        else:
            return self._basic_processor

    def _init_data_dict(self):
        """ details """
        data_dict = {
            "images": [],
            "classes": [],
            "annotations": []
        }
        return data_dict

    # -------------------------------------------
    # label generation and supporting
    # -------------------------------------------     
    def generate_labeles(self):
        """ Details """
        for file in os.listdir(self.img_dir):
            # load image
            img_pth = os.path.join(self.img_dir, file)
            img = cv2.imread(img_pth)

            # get predictions
            output = self.predictor(img)
            preds = output["instances"].to("cpu")

            # get labels from prediction processing
            masks, boxes, classes = self.processor(preds, img)

            # format and store data
            self._store_generated_data(file, img, masks, boxes, classes)

    def _interactive_processor(self, preds, img):
        """ Details """
        # get classes, masks, boxes, and classes        
        scrs = preds.scores.detach().numpy()
        msks = preds.pred_masks.detach().numpy()
        bbxs = preds.pred_boxes.tensor.detach().numpy()
        clss = preds.pred_classes.detach().numpy()

        # init class confidence filtered lists
        if_msks, if_bbxs, if_clss = [], [], []

        # filter by interactive assessment
        for i in range(len(scrs)):
            # filter out low confidence results
            if scrs[i] > 0.5:
                continue

            # get current masks
            msk = msks[i]

            sum = np.sum(msk)
            if sum < 50:
                continue

            # do interactive evaluation
            keep = self._interactive_eval(msk, img)
            if not keep:
                continue

            # keep suitabable masks
            if_msks.append(msk)
            if_bbxs.append(bbxs[i])
            if_clss.append(clss[i])
        
        return if_msks, if_bbxs, if_clss

    def _basic_processor(self, preds, image):
        """ Details """
        # get classes, masks, boxes, and classes
        scrs = preds.scores.detach().numpy()
        msks = preds.pred_masks.detach().numpy()
        bbxs = preds.pred_boxes.tensor.detach().numpy()
        clss = preds.pred_classes.detach().numpy()

        # init class confidence filtered lists
        ccf_msks, ccf_bbxs, ccf_clss = [], [], []
        
        # filter by class confidence
        for i in range(len(scrs)):
            if scrs[i] < self.class_confidence_thresh:
                continue
            ccf_msks.append(msks[i])
            ccf_bbxs.append(bbxs[i])
            ccf_clss.append(clss[i])
        
        # return processed scores, masks, boxes, and classes
        return ccf_msks, ccf_bbxs, ccf_clss
    
    def _interactive_eval(self, mask, image):
        """ Details """
        # overlay detectron2 prediction masks over cv2 image
        coloured_mask = np.zeros_like(image)
        coloured_mask[mask > 0] = [0, 255, 0]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay_image = cv2.addWeighted(
            image_rgb, 
            0.7,
            coloured_mask,
            0.3,
            0
        )

        # present image to user
        overlayed_image_pil = Image.fromarray(overlay_image)
        overlayed_image_pil.save("self_labelling/masked_image.png")

        # wait for keyboard response True if accepted False if not
        print("Press 'a' to keep the mask, 'd' to discard it, or 'q' to quit.")
        user_input = input("Input: ").strip().lower()

        # return true of false based on keyboard response
        if user_input == 'a':
            return True
        if user_input == 'd':
            return False
        if user_input == 'q':
            print("Exit interactive assessment")
            exit()
        else:
            print("Invalid key press, disgarding mask")
            
    def _store_generated_data(self, file, img, masks, boxes, classes):
        """ Details """
        # collect image data
        height, width, _ = img.shape
        image_data = {
            "height": height,
            "width": width,
            "file_name": file 
        }

        # collect annotation data
        polygones = self._mask_to_polygone(masks)
        boxes = self._bbox_processing(boxes)
        classes = self._class_processing(classes)
        annotation_data = {
            "classes": classes,
            "boxes": boxes,
            "masks": polygones,
            "file_name": file
        }

        # add to data dictionary
        self.data_dict["images"].append(image_data)
        self.data_dict["annotations"].append(annotation_data)

    def _class_processing(self, classes):
        """ Details """
        return classes

    def _bbox_processing(self, boxes):
        """ Details """
        return boxes


    def _mask_to_polygone(self, masks):
        """ Details """
        # convert mask list to np array and init all polygones list
        masks = np.array(masks)
        all_polygons = []

        # convert masks to polygones
        for i in range(masks.shape[0]):
            # initialise polygones list
            polygones = [] 

            # get contours
            contours, _ = cv2.findContours(
                masks[i].astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
                )

            # contours to polygons if poly > 4
            for c in contours:
                polygone = c.flatten().tolist()
                if len(polygone) > 4:
                    polygones.append(polygone)
            
            # apppend polygones to all polygones
            all_polygons.append(polygones)
        
        # return all valid polygones
        return all_polygons
            
    # -------------------------------------------
    # exporting
    # -------------------------------------------
    def export_for_anylabeling(self):
        """ Details """
        # do export of data
        for img in self.data_dict["images"]:
            # structure for per image json
            img_dict = {
                "version": "0.3.3",
                "flags": {},
                "shapes": [],
                "imagePath": img["file_name"],
                "imageData": None,
                "imageHeight": img["height"],
                "imageWidth": img["width"],
                "text": ""
            }
            group_counter = 1
            for annos in self.data_dict["annotations"]:
                # only add annotations form matching files
                if annos["file_name"] != img["file_name"]:
                    continue
                
                for anno in annos["masks"]:
                    if len(anno) > 1:
                        for poly in anno:
                            anno_dict = self._annotation_instance_maker(poly, group_id = group_counter)
                            img_dict["shapes"].append(anno_dict)
                        group_counter += 1
                    else:
                        anno_dict = self._annotation_instance_maker(anno[0])
                        img_dict["shapes"].append(anno_dict)
            
            # save image annotation json
            json_file_name = img["file_name"].replace(".jpg", ".json")
            json_path_name = os.path.join(self.out_dir, json_file_name)
            with open(json_path_name, 'w') as json_file:
                json.dump(img_dict, json_file, indent=4)
            
    def _annotation_instance_maker(self, polygon, group_id = None):
        """ Detials """
        # get nested poly 
        nest_poly = [[polygon[i], polygon[i+1]] for i in range(0, len(polygon), 2)]
        
        # convert group id to string if no none
        if group_id:
            group_id = str(group_id)

        # get anno structure
        anno_dict = {
            "label": "jersey_royal",
            "text": "",
            "points": nest_poly,
            "group_id": group_id,
            "shape_type": "polygon",
            "flags": {}
        }

        # return anno stucture
        return anno_dict
    
    def export_as_coco(self):
        """ Detials """
        # do export of data
        pass

# =========================================================
# functions
# =========================================================
def main(cfg):
    """ Details """
    # Initialize LabelGenerator with the given configuration
    label_generator = LabelGenerator(cfg)

    # Generate labels
    label_generator.generate_labeles()

    # Export results (customize as per requirements)
    label_generator.export_for_anylabeling()
    # or label_generator.export_as_coco()

def setup(cfg_pth, weights_pth):
    """ Initialise config and ammend based on command line arguments """
    cfg = get_cfg()
    # adding argments to base config to accomodate pseudo labeling
    add_maskformer2_config(cfg)
    add_pseudo_config(cfg)
    cfg.merge_from_file(cfg_pth)  
    cfg.MODEL.WEIGHTS = weights_pth
    return cfg


# =========================================================
# execute
# =========================================================
if __name__ == "__main__":
    # Define configuration (this might come from argparse in a real-world scenario)
    cfg = {
        "img_dir": "self_labelling/images/label_zone",
        "out_dir": "self_labelling/anylabeling_out", 
        "cfg_pth": "configs/pseudo_labeling/config_files/ps_m2f.yaml",
        "weights_pth": "outputs/m2f/burn_in_030/02_continued/burn_in_best_model.pth",
        "interactive": True,  # Change to True for interactive mode
        "class_confidence_thresh": 0.8,
    }

    # Execute main with the provided configuration
    main(cfg)