from PIL import Image
import os
import json

def check_image_load(image_path):
    try:
        with Image.open(image_path) as img:
            img.load()  # Attempt to load the image fully
        return True
    except Exception as e:
        print(f'Error loading {image_path}: {e}')
        return False

dataset_path = 'datasets/summer_school_data/labelled/val/images'
annotation_file = 'datasets/summer_school_data/labelled/val/images/val_annotations.json'
output_annotation_file = 'datasets/summer_school_data/labelled/val/updated_annotations.json'

# Load COCO annotations
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

# List to hold paths of non-existent files
missing_files = []

# Check all images in the dataset
for image_info in coco_data['images']:
    file_path = os.path.join(dataset_path, image_info['file_name'])
    if not os.path.exists(file_path) or not check_image_load(file_path):
        missing_files.append(image_info['file_name'])
        print(image_info['file_name'])

print(f'Found {len(missing_files)} missing or corrupted files.')

# Remove entries from the COCO annotations
for file_name in missing_files:
    coco_data['images'] = [img for img in coco_data['images'] if img['file_name'] != file_name]
    image_ids = [img['id'] for img in coco_data['images'] if img['file_name'] == file_name]
    coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['image_id'] not in image_ids]

# Save the updated annotations
#with open(output_annotation_file, 'w') as f:
#    json.dump(coco_data, f)

print(f'Saved updated annotations to {output_annotation_file}')