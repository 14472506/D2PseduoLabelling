"""
Details
"""
# imports 
import json
import os
from PIL import Image, ExifTags

# functions
def main(source_dir):
    """ 
    Details
    """
    collection_dict = {
        "categories": [],
        "images": [],
        "annotations": []
        }
    
    cat_dict = {"id": 1, 
                "name": "Jersey Royal", 
                "supercategory": "", 
                "color": "#e24b1d", 
                "metadata": {}, 
                "keypoint_colors": []}
    
    for i, file_name in enumerate(os.listdir(source_dir)):
        
        file_path = os.path.join(source_dir, file_name)
        img = Image.open(file_path)
        w, h = img.size 

        img_dict = {
            "id": i,
            "category_ids": [],
            "path": file_path,
            "width": w,
            "height": h,
            "file_name": file_name,
            "annotated": False
        }

        collection_dict["images"].append(img_dict)
    collection_dict["categories"].append(cat_dict)
    
    json_name = "unlabeled.json"
    out_file = os.path.join(source_dir, json_name)
    
    with open(out_file, "w") as f:
        json.dump(collection_dict, f)

#{"id": 1, 
# "name": "Jersey Royal", 
# "supercategory": "", 
# "color": "#e24b1d", 
# "metadata": {}, 
# "keypoint_colors": []}

#{"id": 137, 
# "category_ids": [], 
# "path": "/datasets/lab_3/image_100_jpg.rf.0fe0c24d18fb0ca41da016ab2eccc334.jpg", 
# "width": 640, 
# "height": 480, 
# "file_name": "image_100_jpg.rf.0fe0c24d18fb0ca41da016ab2eccc334.jpg", 
# "annotated": false,
# }

#{"id": 4397, 
# "image_id": 158, 
# "category_id": 1, 
# "segmentation": [[287, 224, 276, 231, 270, 233, 262, 233, 255, 234, 246, 234, 239, 230, 226, 227]], 
# "area": 610, 
# "bbox": [226, 224, 61, 10], 
# "iscrowd": false, 
# "isbbox": false, 
# "color": "#7b20f9", 
# "metadata": {}}

# execute
if __name__ == "__main__":

    # config
    src_dir = "datasets/unlabeled_dataset"

    # main
    main(src_dir)