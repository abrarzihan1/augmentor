import cv2
import numpy as np
import random


def mosaic(images, annotations, output_size=(640, 640), crop_offset=0.15):
    """
    Applies mosaic augmentation by randomly selecting 4, 6, or 9 images from the input.
    Images are cropped around object centers and scaled to fit perfectly in the output size.

    Args:
        images (List[np.ndarray]): List of 9 images as numpy arrays.
        annotations (List[np.ndarray]): List of 9 annotation arrays corresponding to images.
        output_size (tuple): Output image size (width, height).
        crop_offset (float): Max crop offset for random cropping around center.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented image and updated annotations.
    """
    assert len(images) == 9, "Input must contain exactly 9 images."
    assert len(annotations) == 9, "Input must contain exactly 9 annotation sets."

    # Randomly choose how many images to use
    num_images_options = [4, 6, 9]
    num_images = random.choice(num_images_options)

    # Randomly select images and their corresponding annotations
    selected_indices = random.sample(range(9), num_images)
    selected_images = [images[i] for i in selected_indices]
    selected_annotations = [annotations[i] for i in selected_indices]

    # Determine grid dimensions
    if num_images == 4:
        grid_rows, grid_cols = 2, 2
    elif num_images == 6:
        grid_rows, grid_cols = 2, 3
    elif num_images == 9:
        grid_rows, grid_cols = 3, 3

    w, h = output_size

    # Calculate cell size (each image's final size in the mosaic)
    cell_width = w // grid_cols
    cell_height = h // grid_rows

    # Process each selected image: crop around object center and resize
    processed_images = []
    processed_annotations = []

    for img, anns in zip(selected_images, selected_annotations):
        cropped_img, cropped_anns = _crop_around_object_center(img, anns, (cell_width, cell_height))
        processed_images.append(cropped_img)
        processed_annotations.append(cropped_anns)

    # Create mosaic canvas with exact output size
    mosaic_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Calculate positions for each image in the grid
    positions = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x_offset = col * cell_width
            y_offset = row * cell_height
            positions.append((x_offset, y_offset))

    # Place processed images on canvas
    for i in range(num_images):
        x_off, y_off = positions[i]
        # Ensure we don't exceed canvas boundaries
        end_x = min(x_off + cell_width, w)
        end_y = min(y_off + cell_height, h)
        actual_width = end_x - x_off
        actual_height = end_y - y_off

        # Place the image, cropping if necessary to fit exactly
        img_to_place = processed_images[i][:actual_height, :actual_width]
        mosaic_canvas[y_off:end_y, x_off:end_x] = img_to_place

    # Apply random offset cropping to the entire mosaic
    final_image, final_annotations = _apply_random_crop(
        mosaic_canvas, processed_annotations, positions,
        cell_width, cell_height, output_size, crop_offset
    )

    return final_image, final_annotations


def _crop_around_object_center(image, annotations, target_size):
    """
    Crops an image around a random object's center and resizes to target size.
    """
    h_orig, w_orig = image.shape[:2]
    target_w, target_h = target_size

    if len(annotations) == 0:
        # No objects, just resize the entire image
        resized_img = cv2.resize(image, target_size)
        return resized_img, annotations

    # Select a random object to center around
    random_ann = annotations[random.randint(0, len(annotations) - 1)]
    _, center_x_norm, center_y_norm, _, _ = random_ann

    # Convert to pixel coordinates
    center_x = int(center_x_norm * w_orig)
    center_y = int(center_y_norm * h_orig)

    # Calculate crop size to maintain aspect ratio
    aspect_ratio = target_w / target_h
    orig_aspect_ratio = w_orig / h_orig

    if orig_aspect_ratio > aspect_ratio:
        # Original is wider, crop width
        crop_height = h_orig
        crop_width = int(crop_height * aspect_ratio)
    else:
        # Original is taller, crop height
        crop_width = w_orig
        crop_height = int(crop_width / aspect_ratio)

    # Calculate crop boundaries centered on the object
    crop_x1 = max(0, center_x - crop_width // 2)
    crop_y1 = max(0, center_y - crop_height // 2)
    crop_x2 = min(w_orig, crop_x1 + crop_width)
    crop_y2 = min(h_orig, crop_y1 + crop_height)

    # Adjust if crop goes out of bounds
    if crop_x2 - crop_x1 < crop_width:
        crop_x1 = max(0, crop_x2 - crop_width)
    if crop_y2 - crop_y1 < crop_height:
        crop_y1 = max(0, crop_y2 - crop_height)

    # Crop the image
    cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to target size
    resized_img = cv2.resize(cropped_img, target_size)

    # Update annotations
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    updated_annotations = []

    for ann in annotations:
        class_id, x_norm, y_norm, bw_norm, bh_norm = ann

        # Convert to absolute coordinates in original image
        abs_x = x_norm * w_orig
        abs_y = y_norm * h_orig
        abs_bw = bw_norm * w_orig
        abs_bh = bh_norm * h_orig

        # Adjust for crop offset
        new_abs_x = abs_x - crop_x1
        new_abs_y = abs_y - crop_y1

        # Calculate bounding box in cropped image
        x1 = new_abs_x - abs_bw / 2
        y1 = new_abs_y - abs_bh / 2
        x2 = new_abs_x + abs_bw / 2
        y2 = new_abs_y + abs_bh / 2

        # Clip to crop boundaries
        x1 = max(0, min(crop_w, x1))
        y1 = max(0, min(crop_h, y1))
        x2 = max(0, min(crop_w, x2))
        y2 = max(0, min(crop_h, y2))

        # Skip if bounding box is invalid
        if x2 <= x1 or y2 <= y1:
            continue

        # Convert to normalized coordinates in cropped/resized image
        new_center_x = (x1 + x2) / 2 / crop_w
        new_center_y = (y1 + y2) / 2 / crop_h
        new_bw = (x2 - x1) / crop_w
        new_bh = (y2 - y1) / crop_h

        updated_annotations.append([int(class_id), new_center_x, new_center_y, new_bw, new_bh])

    return resized_img, np.array(updated_annotations)


def _apply_random_crop(mosaic_image, all_annotations, positions, cell_width, cell_height, output_size, crop_offset):
    """
    Applies random cropping to the entire mosaic and updates annotations accordingly.
    """
    mosaic_h, mosaic_w = mosaic_image.shape[:2]
    out_w, out_h = output_size

    # Apply minimal random offset for augmentation
    max_offset_x = int(crop_offset * out_w * 0.1)  # Small offset since images already fit
    max_offset_y = int(crop_offset * out_h * 0.1)

    offset_x = random.randint(-max_offset_x, max_offset_x)
    offset_y = random.randint(-max_offset_y, max_offset_y)

    # Create slightly larger canvas to allow for offset
    padded_canvas = np.zeros((mosaic_h + 2 * max_offset_y, mosaic_w + 2 * max_offset_x, 3), dtype=np.uint8)
    padded_canvas[max_offset_y:max_offset_y + mosaic_h, max_offset_x:max_offset_x + mosaic_w] = mosaic_image

    # Crop with offset
    crop_x = max_offset_x + offset_x
    crop_y = max_offset_y + offset_y
    final_image = padded_canvas[crop_y:crop_y + out_h, crop_x:crop_x + out_w]

    # Update annotations with offset
    final_annotations = []
    for i, anns in enumerate(all_annotations):
        if i >= len(positions):
            continue

        cell_x_off, cell_y_off = positions[i]

        for ann in anns:
            class_id, x_norm, y_norm, bw_norm, bh_norm = ann

            # Convert to absolute coordinates in mosaic
            abs_x = x_norm * cell_width + cell_x_off
            abs_y = y_norm * cell_height + cell_y_off
            abs_bw = bw_norm * cell_width
            abs_bh = bh_norm * cell_height

            # Apply offset
            new_abs_x = abs_x - offset_x
            new_abs_y = abs_y - offset_y

            # Calculate bounding box
            x1 = new_abs_x - abs_bw / 2
            y1 = new_abs_y - abs_bh / 2
            x2 = new_abs_x + abs_bw / 2
            y2 = new_abs_y + abs_bh / 2

            # Clip to output boundaries
            x1 = max(0, min(out_w, x1))
            y1 = max(0, min(out_h, y1))
            x2 = max(0, min(out_w, x2))
            y2 = max(0, min(out_h, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # Convert back to normalized coordinates
            new_center_x = (x1 + x2) / 2 / out_w
            new_center_y = (y1 + y2) / 2 / out_h
            new_bw = (x2 - x1) / out_w
            new_bh = (y2 - y1) / out_h

            final_annotations.append([int(class_id), new_center_x, new_center_y, new_bw, new_bh])

    return final_image, np.array(final_annotations)