import cv2
import numpy as np
import random


def mosaic(images, annotations, output_size=(640, 640)):
    """
    Creates a mosaic by randomly selecting 4, 6, or 9 images from a list of 9.
    Each tile in the mosaic is a crop centered on a random object from its source image.

    Args:
        images (List[np.ndarray]): A list of 9 input images as numpy arrays.
        annotations (List[np.ndarray]): A list of 9 annotation arrays.
                                       Each annotation is [class_id, cx, cy, w, h] in normalized format.
        output_size (tuple): The final size of the mosaic image (width, height).

    Returns:
        Tuple[np.ndarray, np.ndarray]: The final mosaic image and its corresponding annotations.
    """
    # Ensure exactly 9 images and annotations are provided
    assert len(images) == 9, "This function requires a list of exactly 9 images."
    assert len(annotations) == 9, "This function requires a list of exactly 9 annotation sets."

    # Randomly choose the number of images to use for the mosaic
    n = random.choice([4, 6, 9])

    # Randomly select n images and their corresponding annotations
    indices = random.sample(range(9), n)
    selected_images = [images[i] for i in indices]
    selected_annotations = [annotations[i] for i in indices]

    output_w, output_h = output_size

    # Determine the grid layout based on the number of selected images
    if n == 4:
        grid_w, grid_h = 2, 2
    elif n == 6:
        grid_w, grid_h = 3, 2
    else:  # n == 9
        grid_w, grid_h = 3, 3

    tile_w = output_w // grid_w
    tile_h = output_h // grid_h

    # Initialize the mosaic canvas with a neutral gray color
    mosaic_img = np.full((output_h, output_w, 3), 114, dtype=np.uint8)
    final_annotations = []

    for i in range(n):
        img = selected_images[i]
        anns = selected_annotations[i]
        img_h, img_w, _ = img.shape

        # 1. Select a center point for the crop
        if len(anns) > 0:
            # Randomly choose an object to center the crop on
            center_ann = random.choice(anns)
            _, cx_norm, cy_norm, _, _ = center_ann
            center_x, center_y = int(cx_norm * img_w), int(cy_norm * img_h)
        else:
            # If no annotations, crop from the image center
            center_x, center_y = img_w // 2, img_h // 2

        # 2. Define the crop box based on the tile size
        x1_crop = center_x - tile_w // 2
        y1_crop = center_y - tile_h // 2

        # 3. Handle image boundaries and padding
        # Determine the part of the image to be cropped
        x1_img_src = max(x1_crop, 0)
        y1_img_src = max(y1_crop, 0)
        x2_img_src = min(x1_crop + tile_w, img_w)
        y2_img_src = min(y1_crop + tile_h, img_h)

        cropped_img = img[y1_img_src:y2_img_src, x1_img_src:x2_img_src]

        # Determine where to place the crop in the padded tile
        pad_left = -min(0, x1_crop)
        pad_top = -min(0, y1_crop)

        padded_tile = np.full((tile_h, tile_w, 3), 114, dtype=np.uint8)
        padded_tile[pad_top:pad_top + cropped_img.shape[0], pad_left:pad_left + cropped_img.shape[1]] = cropped_img

        # 4. Place the completed tile into the main mosaic canvas
        tile_col = i % grid_w
        tile_row = i // grid_w
        x_offset_mosaic = tile_col * tile_w
        y_offset_mosaic = tile_row * tile_h
        mosaic_img[y_offset_mosaic:y_offset_mosaic + tile_h, x_offset_mosaic:x_offset_mosaic + tile_w] = padded_tile

        # 5. Adjust and clip annotations for the new mosaic
        for ann in anns:
            class_id, cx_n, cy_n, w_n, h_n = ann

            # Absolute coordinates in the original image
            abs_cx, abs_cy = cx_n * img_w, cy_n * img_h
            abs_w, abs_h = w_n * img_w, h_n * img_h
            abs_x1, abs_y1 = abs_cx - abs_w / 2, abs_cy - abs_h / 2

            # Clip bounding box to the cropped tile area
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

