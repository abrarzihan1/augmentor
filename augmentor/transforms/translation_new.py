import numpy as np
import cv2


def translation(image, annotations, max_translate_percent=0.2, p=1.0):
    """
    Applies a random translation to an image and its YOLO annotations with a
    given probability.

    Args:
        image (np.ndarray): The input image in HWC format.
        annotations (np.ndarray): A NumPy array of annotations, where each
                                  row is [class_id, x_center, y_center, width, height].
        max_translate_percent (float): The maximum percentage of the image's
                                       width/height to translate by. Defaults to 0.2.
        p (float): The probability of applying the translation. Must be
                   between 0.0 and 1.0. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the (possibly) translated image and updated annotations.
    """
    # 1. Decide whether to apply the transformation based on probability p
    if np.random.rand() > p:
        return image, annotations

    h, w = image.shape[:2]

    # 2. Generate random translation amounts in pixels
    translate_x_px = int(np.random.uniform(-max_translate_percent, max_translate_percent) * w)
    translate_y_px = int(np.random.uniform(-max_translate_percent, max_translate_percent) * h)

    # 3. Translate the image using OpenCV
    M = np.float32([
        [1, 0, translate_x_px],
        [0, 1, translate_y_px]
    ])
    translated_image = cv2.warpAffine(image, M, (w, h))

    # 4. Translate and clip the bounding boxes
    if annotations is None or annotations.size == 0:
        return translated_image, np.array([])

    if annotations.ndim == 1:
        annotations = np.expand_dims(annotations, axis=0)

    new_annotations = []
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann

        # Denormalize to pixel coordinates
        abs_x_center = x_center * w
        abs_y_center = y_center * h
        abs_width = width * w
        abs_height = height * h
        x_min, y_min = abs_x_center - abs_width / 2, abs_y_center - abs_height / 2
        x_max, y_max = abs_x_center + abs_width / 2, abs_y_center + abs_height / 2

        # Apply pixel translation
        new_x_min, new_y_min = x_min + translate_x_px, y_min + translate_y_px
        new_x_max, new_y_max = x_max + translate_x_px, y_max + translate_y_px

        # Clip boxes to image dimensions
        new_x_min = np.clip(new_x_min, 0, w)
        new_y_min = np.clip(new_y_min, 0, h)
        new_x_max = np.clip(new_x_max, 0, w)
        new_y_max = np.clip(new_y_max, 0, h)

        # Check for valid area
        if new_x_max > new_x_min and new_y_max > new_y_min:
            # Re-calculate and normalize
            new_abs_width = new_x_max - new_x_min
            new_abs_height = new_y_max - new_y_min
            new_x_center = (new_x_min + new_abs_width / 2) / w
            new_y_center = (new_y_min + new_abs_height / 2) / h
            new_width = new_abs_width / w
            new_height = new_abs_height / h

            new_annotations.append([class_id, new_x_center, new_y_center, new_width, new_height])

    return translated_image, np.array(new_annotations)
