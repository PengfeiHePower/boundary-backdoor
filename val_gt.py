import os
import shutil

# Path to the validation images and the class mapping file
val_images_path = '/mnt/scratch/hepengf1/imagenet_1k/val/'
class_mapping_file = '/mnt/scratch/hepengf1/imagenet_1k/ILSVRC2012_validation_ground_truth.txt'

# Read the class mapping file
with open(class_mapping_file, 'r') as f:
    class_mappings = f.readlines()

# Iterate over each file in the validation directory
for idx, filename in enumerate(os.listdir(val_images_path)):
    if filename.endswith(".JPEG"):
        # Find the corresponding class for this file
        class_id = class_mappings[idx].strip()

        # Create a directory for this class if it doesn't exist
        class_dir = os.path.join(val_images_path, class_id)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Move the file to the class directory
        src_file = os.path.join(val_images_path, filename)
        dst_file = os.path.join(class_dir, filename)
        shutil.move(src_file, dst_file)