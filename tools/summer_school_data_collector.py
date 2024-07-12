import os
import shutil

def collate_images_and_txt_files(source_directory, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.png'):
                print(file)
                image_path = os.path.join(root, file)
                txt_path = os.path.join(root, file.replace('.png', '.txt'))
                
                if os.path.exists(txt_path):
                    shutil.copy(image_path, os.path.join(destination_directory, file))
                    shutil.copy(txt_path, os.path.join(destination_directory, os.path.basename(txt_path)))
                else:
                    print(f"Warning: No corresponding txt file for {image_path}")


# Example usage
source_directory = 'datasets/summer_school_data/labelled/improved_gen_test_split'
destination_directory = 'datasets/summer_school_data/labelled/test/'
collate_images_and_txt_files(source_directory, destination_directory)
