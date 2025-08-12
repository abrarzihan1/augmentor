import cv2
import numpy as np
import random


def mosaic(
        images,
        annotations,
        class_freqs,
        extra_images=None,
        extra_annotations=None,
        copy_paste_prob=0.5,
        output_size=(640, 640)
):
    """
    Creates a class-balanced, object-centric mosaic with optional copy-paste augmentation.

    Args:
        images (List[np.ndarray]): List of all images in the dataset.
        annotations (List[np.ndarray]): List of all annotation arrays for the dataset.
        class_freqs (Dict[int, int]): Dictionary mapping class IDs to their frequencies.
        extra_images (List[np.ndarray], optional): Images to use for copy-paste. Defaults to None.
        extra_annotations (List[np.ndarray], optional): Annotations for the extra images. Defaults to None.
        copy_paste_prob (float): Probability of applying copy-paste augmentation.
        output_size (tuple): The final size of the mosaic image (width, height).

    Returns:
        Tuple[np.ndarray, np.ndarray]: The final mosaic image and its corresponding annotations.
    """
    # 1. Randomly choose the number of images for the mosaic
    n = random.choice([4, 6, 9])

    # 2. Sample images inversely by class frequency
    weights = []
    for anns in annotations:
        if len(anns) > 0:
            w = np.mean([1.0 / (class_freqs.get(int(a[0]), 1) + 1e-6) for a in anns])
        else:
            w = 1.0
        weights.append(w)

    weights = np.array(weights) / np.sum(weights)

    num_available_images = len(images)
    replace = num_available_images < n
    selected_indices = np.random.choice(num_available_images, size=n, replace=replace, p=weights)

    selected_images = [images[i] for i in selected_indices]
    selected_annotations = [annotations[i] for i in selected_indices]

    output_w, output_h = output_size

    # 3. Determine the grid layout
    if n == 4:
        grid_w, grid_h = 2, 2
    elif n == 6:
        grid_w, grid_h = (3, 2) if random.random() < 0.5 else (2, 3)
    else:  # n == 9
        grid_w, grid_h = 3, 3

    tile_w = output_w // grid_w
    tile_h = output_h // grid_h

    mosaic_img = np.full((output_h, output_w, 3), 114, dtype=np.uint8)
    final_annotations = []

    # 4. Build the mosaic with object-centric crops
    for i in range(n):
        img, anns = selected_images[i], selected_annotations[i]
        img_h, img_w, _ = img.shape

        if len(anns) > 0:
            center_ann = random.choice(anns)
            _, cx_norm, cy_norm, _, _ = center_ann
            center_x, center_y = int(cx_norm * img_w), int(cy_norm * img_h)
        else:
            center_x, center_y = img_w // 2, img_h // 2

        x1_crop = center_x - tile_w // 2
        y1_crop = center_y - tile_h // 2

        x1_img_src, y1_img_src = max(x1_crop, 0), max(y1_crop, 0)
        x2_img_src, y2_img_src = min(x1_crop + tile_w, img_w), min(y1_crop + tile_h, img_h)

        cropped_img = img[y1_img_src:y2_img_src, x1_img_src:x2_img_src]

        pad_left, pad_top = -min(0, x1_crop), -min(0, y1_crop)

        padded_tile = np.full((tile_h, tile_w, 3), 114, dtype=np.uint8)
        padded_tile[pad_top:pad_top + cropped_img.shape[0], pad_left:pad_left + cropped_img.shape[1]] = cropped_img

        tile_col, tile_row = i % grid_w, i // grid_w
        x_offset_mosaic, y_offset_mosaic = tile_col * tile_w, tile_row * tile_h
        mosaic_img[y_offset_mosaic:y_offset_mosaic + tile_h, x_offset_mosaic:x_offset_mosaic + tile_w] = padded_tile

        for ann in anns:
            class_id, cx_n, cy_n, w_n, h_n = ann
            abs_cx, abs_cy = cx_n * img_w, cy_n * img_h
            abs_w, abs_h = w_n * img_w, h_n * img_h
            abs_x1, abs_y1 = abs_cx - abs_w / 2, abs_cy - abs_h / 2

            new_x1 = max(0, abs_x1 - x1_crop)
            new_y1 = max(0, abs_y1 - y1_crop)
            new_x2 = min(tile_w, abs_x1 + abs_w - x1_crop)
            new_y2 = min(tile_h, abs_y1 + abs_h - y1_crop)

            new_w, new_h = new_x2 - new_x1, new_y2 - new_y1

            if new_w > 0 and new_h > 0:
                final_cx = new_x1 + new_w / 2 + x_offset_mosaic
                final_cy = new_y1 + new_h / 2 + y_offset_mosaic
                final_annotations.append([
                    class_id,
                    final_cx / output_w, final_cy / output_h,
                    new_w / output_w, new_h / output_h
                ])

    # 5. Apply Copy-Paste Augmentation
    if extra_images and extra_annotations and random.random() < copy_paste_prob:
        # Select a random source image and annotation from the extra pool
        source_idx = random.randrange(len(extra_images))
        source_img = extra_images[source_idx]
        source_anns = extra_annotations[source_idx]

        if len(source_anns) > 0:
            # Select a random object to copy
            obj_ann = random.choice(source_anns)
            class_id, cx_n, cy_n, w_n, h_n = obj_ann

            src_h, src_w, _ = source_img.shape

            # Get absolute coordinates of the object in the source image
            abs_w, abs_h = int(w_n * src_w), int(h_n * src_h)
            abs_x1 = int((cx_n * src_w) - (abs_w / 2))
            abs_y1 = int((cy_n * src_h) - (abs_h / 2))

            # Ensure coordinates are valid and extract the patch
            if abs_x1 >= 0 and abs_y1 >= 0 and abs_w > 0 and abs_h > 0:
                patch = source_img[abs_y1:abs_y1 + abs_h, abs_x1:abs_x1 + abs_w]

                # Find a random valid location to paste the patch
                if patch.shape[0] < output_h and patch.shape[1] < output_w:
                    paste_x = random.randint(0, output_w - patch.shape[1])
                    paste_y = random.randint(0, output_h - patch.shape[0])

                    # Paste the object onto the mosaic
                    mosaic_img[paste_y:paste_y + patch.shape[0], paste_x:paste_x + patch.shape[1]] = patch

                    # Add the new annotation
                    new_cx = (paste_x + patch.shape[1] / 2) / output_w
                    new_cy = (paste_y + patch.shape[0] / 2) / output_h
                    new_w = patch.shape[1] / output_w
                    new_h = patch.shape[0] / output_h

                    final_annotations.append([class_id, new_cx, new_cy, new_w, new_h])

    return mosaic_img, np.array(final_annotations)

