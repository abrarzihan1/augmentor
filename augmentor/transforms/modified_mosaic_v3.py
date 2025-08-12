import cv2
import numpy as np
import random


def mosaic(
        images,
        annotations,
        class_freqs,
        output_size=(640, 640)
):
    """
    Creates a class-balanced, object-centric mosaic from a dataset.

    It randomly selects 4, 6, or 9 images, giving priority to images with
    rarer classes. Each tile in the mosaic is a crop centered on a random
    object from its source image.

    Args:
        images (List[np.ndarray]): List of all images in the dataset.
        annotations (List[np.ndarray]): List of all annotation arrays for the dataset.
                                           Each annotation is [class_id, cx, cy, w, h].
        class_freqs (Dict[int, int]): A dictionary mapping class IDs to their frequencies.
        output_size (tuple): The final size of the mosaic image (width, height).

    Returns:
        Tuple[np.ndarray, np.ndarray]: The final mosaic image and its annotations.
    """
    # 1. Randomly choose the number of images for the mosaic
    n = random.choice([4, 6, 9])

    # 2. Sample images inversely by class frequency
    weights = []
    # Iterate through all annotations in the dataset to calculate weights
    for anns in annotations:
        if len(anns) > 0:
            # Weight is the mean inverse frequency of classes in the image
            # A small epsilon is added to avoid division by zero for unknown classes
            w = np.mean([1.0 / (class_freqs.get(int(a[0]), 1) + 1e-6) for a in anns])
        else:
            # Assign a default weight for images with no objects
            w = 1.0
        weights.append(w)

    # Normalize weights to form a probability distribution
    weights = np.array(weights) / np.sum(weights)

    # Choose n image indices based on the calculated weights
    num_available_images = len(images)
    # Use replacement if n is greater than the number of available images
    replace = num_available_images < n
    selected_indices = np.random.choice(
        num_available_images, size=n, replace=replace, p=weights
    )

    selected_images = [images[i] for i in selected_indices]
    selected_annotations = [annotations[i] for i in selected_indices]

    output_w, output_h = output_size

    # 3. Determine the grid layout
    if n == 4:
        grid_w, grid_h = 2, 2
    elif n == 6:
        # Randomly choose between a 3x2 and 2x3 layout
        grid_w, grid_h = (3, 2) if random.random() < 0.5 else (2, 3)
    else:  # n == 9
        grid_w, grid_h = 3, 3

    tile_w = output_w // grid_w
    tile_h = output_h // grid_h

    # Initialize the mosaic canvas
    mosaic_img = np.full((output_h, output_w, 3), 114, dtype=np.uint8)
    final_annotations = []

    # 4. Build the mosaic with object-centric crops
    for i in range(n):
        img = selected_images[i]
        anns = selected_annotations[i]
        img_h, img_w, _ = img.shape

        # Select a center point for the crop (on an object or image center)
        if len(anns) > 0:
            center_ann = random.choice(anns)
            _, cx_norm, cy_norm, _, _ = center_ann
            center_x, center_y = int(cx_norm * img_w), int(cy_norm * img_h)
        else:
            center_x, center_y = img_w // 2, img_h // 2

        # Define the crop box based on the tile size
        x1_crop = center_x - tile_w // 2
        y1_crop = center_y - tile_h // 2

        # Handle image boundaries by finding the valid crop area
        x1_img_src = max(x1_crop, 0)
        y1_img_src = max(y1_crop, 0)
        x2_img_src = min(x1_crop + tile_w, img_w)
        y2_img_src = min(y1_crop + tile_h, img_h)

        cropped_img = img[y1_img_src:y2_img_src, x1_img_src:x2_img_src]

        # Calculate padding needed for the tile
        pad_left = -min(0, x1_crop)
        pad_top = -min(0, y1_crop)

        # Create a padded tile and place the cropped image into it
        padded_tile = np.full((tile_h, tile_w, 3), 114, dtype=np.uint8)
        padded_tile[pad_top:pad_top + cropped_img.shape[0], pad_left:pad_left + cropped_img.shape[1]] = cropped_img

        # Place the completed tile into the main mosaic canvas
        tile_col = i % grid_w
        tile_row = i // grid_w
        x_offset_mosaic = tile_col * tile_w
        y_offset_mosaic = tile_row * tile_h
        mosaic_img[y_offset_mosaic:y_offset_mosaic + tile_h, x_offset_mosaic:x_offset_mosaic + tile_w] = padded_tile

        # 5. Adjust annotations for objects within the new tile
        for ann in anns:
            class_id, cx_n, cy_n, w_n, h_n = ann

            # Convert normalized annotation to absolute pixel values
            abs_cx, abs_cy = cx_n * img_w, cy_n * img_h
            abs_w, abs_h = w_n * img_w, h_n * img_h
            abs_x1, abs_y1 = abs_cx - abs_w / 2, abs_cy - abs_h / 2

            # Clip the bounding box to the boundaries of the cropped tile
            new_x1 = max(0, abs_x1 - x1_crop)
            new_y1 = max(0, abs_y1 - y1_crop)
            new_x2 = min(tile_w, abs_x1 + abs_w - x1_crop)
            new_y2 = min(tile_h, abs_y1 + abs_h - y1_crop)

            new_w, new_h = new_x2 - new_x1, new_y2 - new_y1

            if new_w > 0 and new_h > 0:
                # Calculate new center and dimensions relative to the final mosaic
                final_cx = new_x1 + new_w / 2 + x_offset_mosaic
                final_cy = new_y1 + new_h / 2 + y_offset_mosaic

                # Normalize coordinates for the final output
                final_cx_norm = final_cx / output_w
                final_cy_norm = final_cy / output_h
                final_w_norm = new_w / output_w
                final_h_norm = new_h / output_h

                final_annotations.append([class_id, final_cx_norm, final_cy_norm, final_w_norm, final_h_norm])

    return mosaic_img, np.array(final_annotations)

