import cv2
import random
import os
import numpy as np
from augmentor import augment, utils

random.seed(1)
np.random.seed(1)

image_dir = '../datasets/tomato/train/images/'
annotation_dir = '../datasets/tomato/train/labels/'

output_img_dir = './output_1/images/'
output_annotation_dir = './output_1/labels/'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for i in range(len(image_files)):
    selected_files = random.sample(image_files, 4)
    images = [cv2.imread(os.path.join(image_dir, f)) for f in selected_files]
    annotations = [utils.load_yolo_annotation(os.path.join(annotation_dir, f.replace('.jpg', '.txt'))) for f in selected_files]

    aug_image, aug_annotations = augment.modified_mosaic(images, annotations)

    augmented_image_filename = f"aug_{i + 1}.jpg"
    augmented_image_path = os.path.join(output_img_dir, augmented_image_filename)
    cv2.imwrite(augmented_image_path, aug_image)

    augmented_annotation_filename = f"aug_{i + 1}.txt"
    augmented_annotation_path = os.path.join(output_annotation_dir, augmented_annotation_filename)
    utils.save_yolo_annotation(augmented_annotation_path, aug_annotations)

utils.copy_folder_contents(image_dir, output_img_dir)
utils.copy_folder_contents(annotation_dir, output_annotation_dir)
