import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from augmentor import augment, utils

seeds = [1]

def draw_image(image, bboxes, class_names=None, color=(0, 255, 0), thickness=2):
    """
    Draw YOLO bounding boxes on an image.

    Parameters:
    - image (np.ndarray): Input image (H, W, C) in BGR format.
    - bboxes (np.ndarray or list): Bounding boxes in YOLO format (class_id, x_center, y_center, width, height),
      normalized between 0 and 1.
    - class_names (list, optional): List of class names indexed by class_id. If provided, labels_o will be drawn.
    - color (tuple): Bounding box color in BGR (default green).
    - thickness (int): Thickness of bounding box lines.

    Returns:
    - img_out (np.ndarray): Copy of the image with bounding boxes drawn.
    """
    img_out = image.copy()
    h, w = img_out.shape[:2]

    for bbox in bboxes:
        class_id, x_c, y_c, bw, bh = bbox

        # Convert normalized coordinates to pixel values
        x_c *= w
        y_c *= h
        bw *= w
        bh *= h

        # Calculate box corners
        x1 = int(x_c - bw / 2)
        y1 = int(y_c - bh / 2)
        x2 = int(x_c + bw / 2)
        y2 = int(y_c + bh / 2)

        # Draw rectangle
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness)

        # Draw label if class_names provided
        if class_names is not None:
            label = str(class_names[int(class_id)])
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_out, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
            cv2.putText(img_out, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_out

# Define paths for input images and annotations
image_dir = '../datasets/tomato/train/images/'
annotation_dir = '../datasets/tomato/train/labels/'

# Create a list of all JPEG files in the image directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Loop through each seed to create different sets of augmentations
for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)

    # Determine output directories based on the current seed
    if seed == 1:
        output_img_dir = './aug_1/images'
        output_annotation_dir = './aug_1/labels'
    elif seed == 42:
        output_img_dir = './aug_2/images'
        output_annotation_dir = './aug_2/labels'
    elif seed == 99:
        output_img_dir = './aug_3/images'
        output_annotation_dir = './aug_3/labels'
    else:
        # Skip seeds that don't have a corresponding output directory defined
        continue

    # Create the output directories if they don't already exist
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)

    # Loop through each image file for augmentation
    for image_file in tqdm(image_files):
        # Construct the full path to the image and its corresponding annotation file
        image_path = os.path.join(image_dir, image_file)
        annotation_path = os.path.join(annotation_dir, image_file.replace('.jpg', '.txt'))

        # Read the image and load its annotation data
        image = cv2.imread(image_path)
        annotation = utils.load_yolo_annotation(annotation_path)

        # Perform augmentation twice
        for i in range(1):
            # The random seed is set at the start of the outer loop, so if `augment.brightness_contrast`
            # internally uses randomness, it will produce different results on each call.
            # If the augmentation function is deterministic, you might need to introduce
            # randomness here if you want varied augmentations for the same input image.

            aug_img, aug_labels = augment.rotation(image, annotation)  # Your augmentation function
            aug_img = draw_image(aug_img, aug_labels)

            # Split the base name and extension
            base_name, ext = os.path.splitext(image_file)

            # Compose filenames with suffix
            output_image_filename = f"{base_name}_aug_{i + 1}{ext}"
            output_annotation_filename = f"{base_name}_aug_{i + 1}.txt"

            # Save the augmented image and its annotation
            cv2.imwrite(os.path.join(output_img_dir, output_image_filename), aug_img)
            utils.save_yolo_annotation(os.path.join(output_annotation_dir, output_annotation_filename), aug_labels)

    # Copy original files to the augmentation directories after processing
    utils.copy_folder_contents(image_dir, output_img_dir)
    utils.copy_folder_contents(annotation_dir, output_annotation_dir)