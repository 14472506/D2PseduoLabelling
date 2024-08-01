import json
import random
from collections import defaultdict

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def read_txt(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def ensure_class_coverage(annotations, categories):
    class_coverage = defaultdict(list)
    for ann in annotations:
        class_coverage[ann['category_id']].append(ann)
    
    ensured_annotations = []
    for cat_id, anns in class_coverage.items():
        ensured_annotations.append(random.choice(anns))
    
    return ensured_annotations

def reduce_train_json(train_json, core_images, reduction_percentage):
    # Extract the list of images and annotations from the train JSON
    train_images = train_json['images']
    train_annotations = train_json['annotations']
    categories = train_json['categories']
    
    # Separate the core images and other images
    core_set = set(core_images)
    core_train_images = [img for img in train_images if img['file_name'] in core_set]
    other_train_images = [img for img in train_images if img['file_name'] not in core_set]
    
    # Calculate the number of images to retain
    total_images_to_retain = int(len(train_images) * (1 - reduction_percentage))
    other_images_to_retain = total_images_to_retain - len(core_train_images)
    
    # Randomly select the other images to retain
    if other_images_to_retain > 0:
        retained_other_images = random.sample(other_train_images, other_images_to_retain)
    else:
        retained_other_images = []
    
    # Combine the core images with the randomly selected other images
    reduced_train_images = core_train_images + retained_other_images
    reduced_image_ids = set(img['id'] for img in reduced_train_images)
    
    # Filter annotations to only include those related to the retained images
    reduced_train_annotations = [ann for ann in train_annotations if ann['image_id'] in reduced_image_ids]

    # Ensure at least one instance of each class is included
    ensured_annotations = ensure_class_coverage(reduced_train_annotations, categories)
    ensured_image_ids = set(ann['image_id'] for ann in ensured_annotations)
    
    final_reduced_images = [img for img in reduced_train_images if img['id'] in ensured_image_ids]
    final_reduced_annotations = ensured_annotations + [
        ann for ann in reduced_train_annotations if ann['image_id'] not in ensured_image_ids
    ]
    
    # Ensure that images related to ensured annotations are included in the final images list
    final_image_ids = set(img['id'] for img in final_reduced_images)
    final_reduced_images += [img for img in reduced_train_images if img['id'] not in final_image_ids and img['id'] in ensured_image_ids]
    
    # Update the train JSON with the reduced list of images and annotations
    train_json['images'] = final_reduced_images
    train_json['annotations'] = final_reduced_annotations

    return train_json

if __name__ == "__main__":
    train_json_path = "datasets/coco2017/annotations/instances_train2017.json"
    core_images_txt_path = None
    output_json_path = "datasets/coco2017/annotations/instances_train2017_05perc.json"
    reduction_percentage = 0.95  # Reduction percentage

    train_json_data = read_json(train_json_path)
    core_images_list = read_txt(core_images_txt_path) if core_images_txt_path else []

    reduced_train_json_data = reduce_train_json(train_json_data, core_images_list, reduction_percentage)

    write_json(output_json_path, reduced_train_json_data)

