import os
import json
from PIL import Image

def convert_to_coco(source_directory, destination_json):
    images = []
    annotations = []
    categories = []

    # Assuming a single category for simplicity
    categories.append(
        {"id": 1, "name": "crop"})
    categories.append(
        {"id": 2, "name": "weed"})
    categories.append(
        {"id": 3, "name": "partial_crop1"})
    categories.append(
        {"id": 4, "name": "partial_crop2"})

    annotation_id = 0
    image_id = 0

    for file in os.listdir(source_directory):
        if file.endswith('.png'):
            image_path = os.path.join(source_directory, file)
            txt_path = os.path.join(source_directory, file.replace('.png', '.txt'))
            
            if not os.path.exists(txt_path):
                print(f"Warning: No corresponding txt file for {image_path}")
                continue
            
            # Read the image to get its width and height
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Add image information to the images list
            images.append({
                "id": image_id,
                "file_name": file,
                "width": width,
                "height": height
            })
            
            # Read the bounding box information from the txt file
            print(file)
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Incorrect format in file {txt_path}")
                        continue
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                    class_id = int(class_id) + 1
                    print(class_id)
                    # Convert YOLO format to COCO format
                    x_center *= width
                    y_center *= height
                    bbox_width *= width
                    bbox_height *= height
                    
                    x = x_center - bbox_width / 2
                    y = y_center - bbox_height / 2
                    
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            image_id += 1
    
    # Create the final COCO JSON structure
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to the destination JSON file
    with open(destination_json, 'w') as json_file:
        json.dump(coco_output, json_file, indent=4)

# Example usage
source_directory = 'datasets/summer_school_data/labelled/test'
destination_json = 'datasets/summer_school_data/labelled/test/test_annotations.json'
convert_to_coco(source_directory, destination_json)