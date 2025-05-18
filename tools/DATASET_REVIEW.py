#!/usr/bin/env python3
import json
import shutil
import os
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python classify_coco.py <path_to_coco_json> <images_directory>")
        sys.exit(1)

    coco_json_path = sys.argv[1]
    images_dir = sys.argv[2]
    output_image = "current_image.jpg"  # This is the file that will be updated each iteration.

    # Load the COCO JSON file.
    try:
        with open(coco_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

    # Check that the JSON has an 'images' key.
    if "images" not in data:
        print("The provided JSON file does not contain an 'images' key.")
        sys.exit(1)

    # Dictionary to store classifications: using image IDs as keys.
    classifications = {}

    print("Starting classification. Type 'q' to quit at any time.\n")

    # Iterate over each image record.
    for image in data["images"]:
        file_name = image.get("file_name")
        image_id = image.get("id")
        if not file_name:
            print(f"Image record {image} has no 'file_name' field, skipping.")
            continue

        # Skip images with the phrase "mar" in the title (case-insensitive)
        if "mar" in file_name.lower():
            print(f"Skipping image '{file_name}' (ID: {image_id}) due to phrase 'mar' in title.")
            continue

        # Construct full path to the source image.
        src_path = os.path.join(images_dir, file_name)
        if not os.path.exists(src_path):
            print(f"Image file '{src_path}' not found, skipping.")
            continue

        # Copy the current image to a fixed file name.
        try:
            shutil.copy(src_path, output_image)
            print(f"Loaded image '{file_name}' (ID: {image_id}) as '{output_image}'.")
        except Exception as e:
            print(f"Error copying file '{src_path}' to '{output_image}': {e}")
            continue

        # Prompt user for classification.
        while True:
            classification = input("Enter classification (1, 2, 3, or 4) for this image (or 'q' to quit): ").strip()
            if classification in ['1', '2', '3', '4']:
                classifications[image_id] = classification
                break
            elif classification.lower() == 'q':
                print("Quitting classification early.")
                save_classifications(classifications)
                sys.exit(0)
            else:
                print("Invalid input. Please enter 1, 2, 3, or 4 (or 'q' to quit).")

    # Save final classification results.
    save_classifications(classifications)
    print("All images classified. Results saved in 'classification_results.json'.")

def save_classifications(classifications):
    try:
        with open("classification_results.json", "w") as f:
            json.dump(classifications, f, indent=2)
        print("Classification results saved.")
    except Exception as e:
        print(f"Error saving classification results: {e}")

if __name__ == "__main__":
    main()
