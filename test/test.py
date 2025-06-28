import os
from augmentor import augment, utils
import cv2
import numpy as np
import random

random.seed(1)
np.random.seed(1)

image_dir = '../datasets/tomato/train/images/'
annotation_dir = '../datasets/tomato/train/labels/'

output_img_dir = './output/images/'
output_annotation_dir = './output/labels/'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in image_files:
    print(f'Augmenting {image_file}')
    image = cv2.imread(f'{image_dir}/{image_file}')
    annotation_file = image_file.replace('.jpg', '.txt')
    annotation_file = os.path.join(annotation_dir, annotation_file)
    annotation = utils.load_yolo_annotation(annotation_file)

    aug_img, aug_labels = augment.cutout(image, annotation)

    cv2.imwrite(f'{output_img_dir}/aug_{image_file}', aug_img)
    utils.save_yolo_annotation(f"{output_annotation_dir}/aug_{image_file.replace('.jpg', '.txt')}", aug_labels)

utils.copy_folder_contents(image_dir, output_img_dir)
utils.copy_folder_contents(annotation_dir, output_annotation_dir)