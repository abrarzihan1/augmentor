import numpy as np
import cv2
import os
import shutil
import random
from . import augment

# Helper function to load YOLO annotations (text format)
def load_yolo_annotation(annotation_path):
    if not os.path.exists(annotation_path):
        return np.array([])  # Return an empty array if the annotation file doesn't exist
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    if not lines:
        return np.array([])  # Return an empty array if the annotation file is empty
    annotations = [list(map(float, line.strip().split())) for line in lines]
    return np.array(annotations)

# Helper function to save YOLOv8 annotations
def save_yolo_annotation(annotation_path, annotations):
    with open(annotation_path, 'w') as file:
        for ann in annotations:
            file.write(' '.join(map(str, ann)) + '\n')

def resize_image(image_array, desired_shape=(640, 640), channels=None):
    """
    Resizes a NumPy array image to the desired shape and optionally changes the number of channels.

    Parameters:
    - image_array (numpy.ndarray): The image to resize.
    - desired_shape (tuple): Desired shape as (height, width).
    - channels (int, optional): Desired number of channels (1, 3, or 4). If None, keep original.

    Returns:
    - resized_image (numpy.ndarray): Image resized to the desired shape and channels.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if len(desired_shape) != 2:
        raise ValueError("desired_shape must be a tuple of (height, width).")

    # Resize image (OpenCV uses (width, height) order)
    resized_image = cv2.resize(image_array, (desired_shape[1], desired_shape[0]))

    # Convert number of channels if needed
    if channels is not None:
        if channels == 1:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        elif channels == 3:
            if len(resized_image.shape) == 2:  # Grayscale to BGR
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
            elif resized_image.shape[2] == 4:  # RGBA to BGR
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
        elif channels == 4:
            if resized_image.shape[2] == 3:  # BGR to BGRA
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2BGRA)
            elif len(resized_image.shape) == 2:  # Grayscale to BGRA
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGRA)
        else:
            raise ValueError("Unsupported number of channels. Use 1, 3, or 4.")

    return resized_image

def normalize_image(image_array):
    """
    Normalizes a NumPy image array.

    Parameters:
    - image_array (numpy.ndarray): The image to normalize.

    Returns:
    - normalized_image (numpy.ndarray): The normalized image.
    """
    image = image_array.astype(np.float32)
    return image / 255.0

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


def copy_folder_contents(source_folder, destination_folder):
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} does not exist.")
        return

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through the files and subdirectories in the source folder
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        destination_item = os.path.join(destination_folder, item)

        if os.path.isdir(source_item):
            # If it's a directory, copy the entire directory and its contents
            shutil.copytree(source_item, destination_item)
        else:
            # If it's a file, copy the file
            shutil.copy2(source_item, destination_item)

    print(f"All contents from {source_folder} have been copied to {destination_folder}.")


def augment_image(method, image_dir, annotation_dir, output_img_dir, output_annotation_dir, seed):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    if method == "mixup":
        k = 2
    else:
        k = 4

    for i in range(len(image_files)):
        selected_files = random.sample(image_files, k)
        images = [cv2.imread(os.path.join(image_dir, f)) for f in selected_files]
        annotations = [load_yolo_annotation(os.path.join(annotation_dir, f.replace('.jpg', '.txt'))) for f in
                       selected_files]

        aug_image, aug_annotations = augment.apply_transform(method, images, annotations)

        augmented_image_filename = f"aug_{i + 1}.jpg"
        augmented_image_path = os.path.join(output_img_dir, augmented_image_filename)
        cv2.imwrite(augmented_image_path, aug_image)

        augmented_annotation_filename = f"aug_{i + 1}.txt"
        augmented_annotation_path = os.path.join(output_annotation_dir, augmented_annotation_filename)
        save_yolo_annotation(augmented_annotation_path, aug_annotations)

    copy_folder_contents(image_dir, output_img_dir)
    copy_folder_contents(annotation_dir, output_annotation_dir)