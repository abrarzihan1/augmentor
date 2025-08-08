import cv2
import numpy as np
import random
from . import cutout

def mosaic(images, annotations, output_size=(640, 640), crop_offset=0.15,
           apply_cutout=True, cutout_prob=0.3, cutout_kwargs=None):
    """
    Applies mosaic augmentation with optional Cutout to four images.

    Args:
        images (list of np.ndarray): List of 4 images.
        annotations (list of list): List of 4 annotation sets.
        output_size (tuple): Final size of the mosaic (W, H).
        crop_offset (float): Max offset for mosaic crop.
        apply_cutout (bool): Whether to apply Cutout to each image.
        cutout_prob (float): Probability of applying Cutout to each image.
        cutout_kwargs (dict): Parameters for Cutout function.

    Returns:
        cropped_mosaic (np.ndarray): Final mosaic image
        updated_annotations (np.ndarray): Transformed YOLO annotations
    """
    assert len(images) == 4, "Exactly four images are required for mosaic."
    assert len(annotations) == 4, "Exactly four annotation sets are required."

    w, h = output_size
    # Apply random augmentations per image BEFORE resizing
    processed_images = []
    for i in range(len(images)):
        if random.random() < cutout_prob:
            images[i], annotations[i] = cutout.apply(images[i], annotations[i])
        processed_images.append(cv2.resize(images[i], output_size))

    # Mosaic canvas
    mosaic_canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    quadrant_offsets = [(0, 0), (w, 0), (0, h), (w, h)]

    for i, (x_off, y_off) in enumerate(quadrant_offsets):
        mosaic_canvas[y_off:y_off + h, x_off:x_off + w] = processed_images[i]

    # Random crop center
    x_offset_crop = int(random.uniform(-crop_offset, crop_offset) * w)
    y_offset_crop = int(random.uniform(-crop_offset, crop_offset) * h)
    x_center = w + x_offset_crop
    y_center = h + y_offset_crop
    crop_x = x_center - w // 2
    crop_y = y_center - h // 2
    cropped_mosaic = mosaic_canvas[crop_y:crop_y + h, crop_x:crop_x + w]

    # Transform annotations
    updated_annotations = []
    for i, anns in enumerate(annotations):
        quad_x_off, quad_y_off = quadrant_offsets[i]
        for ann in anns:
            class_id, x_norm, y_norm, bw_norm, bh_norm = ann
            abs_x = x_norm * w + quad_x_off
            abs_y = y_norm * h + quad_y_off
            abs_bw = bw_norm * w
            abs_bh = bh_norm * h

            new_abs_x = abs_x - crop_x
            new_abs_y = abs_y - crop_y
            x1 = new_abs_x - abs_bw / 2
            y1 = new_abs_y - abs_bh / 2
            x2 = new_abs_x + abs_bw / 2
            y2 = new_abs_y + abs_bh / 2

            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            new_center_x = (x1 + x2) / 2 / w
            new_center_y = (y1 + y2) / 2 / h
            new_bw = (x2 - x1) / w
            new_bh = (y2 - y1) / h

            updated_annotations.append([int(class_id), new_center_x, new_center_y, new_bw, new_bh])

    return cropped_mosaic, np.array(updated_annotations, dtype=np.float32)
