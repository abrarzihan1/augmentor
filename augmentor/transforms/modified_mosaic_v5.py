import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional


# ---------------------------
# Basic geometric augmentations
# ---------------------------

def random_flip(image: np.ndarray, annotations: np.ndarray, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a random horizontal flip to an image and its annotations.

    Args:
        image (np.ndarray): The input image.
        annotations (np.ndarray): The corresponding annotations.
        prob (float): The probability of applying the flip.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The potentially flipped image and adjusted annotations.
    """
    if random.random() < prob:
        image = np.fliplr(image)
        if len(annotations) > 0:
            # For YOLO format [class, cx, cy, w, h], only cx needs to be adjusted
            annotations[:, 1] = 1.0 - annotations[:, 1]
    return image, annotations


def _build_affine_matrices(
    height: int,
    width: int,
    degrees: float,
    scale: float,
    shear: float
) -> np.ndarray:
    # Create the transformation matrix
    center = (width / 2, height / 2)
    angle = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)

    # Combined rotation and scaling matrix
    rot_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=s)

    # Add shear to the matrix
    shear_x = random.uniform(-shear, shear)
    shear_y = random.uniform(-shear, shear)
    shear_matrix = np.array([
        [1, -np.tan(np.radians(shear_x)), 0],
        [-np.tan(np.radians(shear_y)), 1, 0]
    ])

    # Combine all transformations
    trans_matrix = shear_matrix @ np.vstack([rot_matrix, [0, 0, 1]])
    return trans_matrix[:2]


def _warp_image(image: np.ndarray, M: np.ndarray, width: int, height: int) -> np.ndarray:
    # Apply the affine transformation to the image
    return cv2.warpAffine(image, M, (width, height), borderValue=(114, 114, 114))


def _transform_yolo_boxes(
    annotations: np.ndarray,
    M: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    # Transform bounding box annotations
    new_annotations = []
    for ann in annotations:
        class_id, cx, cy, w, h = ann

        # Get corner points of the bounding box
        box_w, box_h = w * width, h * height
        x1, y1 = (cx * width) - box_w / 2, (cy * height) - box_h / 2
        x2, y2 = x1 + box_w, y1 + box_h
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]).T

        # Apply the transformation to the corner points
        transformed_corners = (M @ corners).T

        # Get the new axis-aligned bounding box
        new_x1 = min(transformed_corners[:, 0])
        new_y1 = min(transformed_corners[:, 1])
        new_x2 = max(transformed_corners[:, 0])
        new_y2 = max(transformed_corners[:, 1])

        # Clip to image boundaries
        new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
        new_x2, new_y2 = min(width, new_x2), min(height, new_y2)

        # Calculate new width, height, and center
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1

        # Filter out boxes that are too small or invalid after transform
        if new_w > 1 and new_h > 1:
            new_cx = (new_x1 + new_w / 2) / width
            new_cy = (new_y1 + new_h / 2) / height
            new_annotations.append([class_id, new_cx, new_cy, new_w / width, new_h / height])

    return np.array(new_annotations)


def random_affine(
    image: np.ndarray,
    annotations: np.ndarray,
    degrees: float = 10,
    scale: float = 0.1,
    shear: float = 10,
    prob: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies random affine transformations (rotation, scale, shear) to an image and its annotations.

    Args:
        image (np.ndarray): Input image.
        annotations (np.ndarray): Annotations for the image.
        degrees (float): Range of random rotation in degrees.
        scale (float): Range of random scaling.
        shear (float): Range of random shear in degrees.
        prob (float): Probability of applying the transformation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The transformed image and annotations.
    """
    if random.random() > prob or len(annotations) == 0:
        return image, annotations

    height, width = image.shape[:2]
    M = _build_affine_matrices(height, width, degrees, scale, shear)
    image = _warp_image(image, M, width, height)
    new_annotations = _transform_yolo_boxes(annotations, M, width, height)
    return image, new_annotations


# ---------------------------
# Mosaic helpers
# ---------------------------

def _choose_mosaic_count() -> int:
    # 1. Randomly choose the number of images for the mosaic
    return random.choice([4, 6, 9])


def _compute_sampling_weights(
    annotations: List[np.ndarray],
    class_freqs: Dict[int, int]
) -> np.ndarray:
    # 2. Sample images inversely by class frequency
    weights = []
    for anns in annotations:
        if len(anns) > 0:
            w = np.mean([1.0 / (class_freqs.get(int(a[0]), 1) + 1e-6) for a in anns])
        else:
            w = 1.0
        weights.append(w)
    weights = np.array(weights)
    weights = weights / np.sum(weights) if weights.sum() > 0 else np.full_like(weights, 1.0 / len(weights))
    return weights


def _select_images_by_weight(
    images: List[np.ndarray],
    annotations: List[np.ndarray],
    n: int,
    weights: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    num_available_images = len(images)
    replace = num_available_images < n
    selected_indices = np.random.choice(num_available_images, size=n, replace=replace, p=weights)
    selected_images = [images[i] for i in selected_indices]
    selected_annotations = [annotations[i] for i in selected_indices]
    return selected_images, selected_annotations, selected_indices


def _grid_layout(n: int) -> Tuple[int, int]:
    # 3. Determine the grid layout
    if n == 4:
        return 2, 2
    if n == 6:
        return (3, 2) if random.random() < 0.5 else (2, 3)
    # n == 9
    return 3, 3


def _object_centric_center(img: np.ndarray, anns: np.ndarray) -> Tuple[int, int]:
    img_h, img_w, _ = img.shape
    if len(anns) > 0:
        center_ann = random.choice(anns)
        _, cx_norm, cy_norm, _, _ = center_ann
        return int(cx_norm * img_w), int(cy_norm * img_h)
    return img_w // 2, img_h // 2


def _compute_crop_coords(center_x: int, center_y: int, tile_w: int, tile_h: int) -> Tuple[int, int]:
    x1_crop = center_x - tile_w // 2
    y1_crop = center_y - tile_h // 2
    return x1_crop, y1_crop


def _crop_and_pad_tile(
    img: np.ndarray,
    x1_crop: int,
    y1_crop: int,
    tile_w: int,
    tile_h: int
) -> Tuple[np.ndarray, int, int, int, int]:
    img_h, img_w, _ = img.shape
    x1_img_src, y1_img_src = max(x1_crop, 0), max(y1_crop, 0)
    x2_img_src, y2_img_src = min(x1_crop + tile_w, img_w), min(y1_crop + tile_h, img_h)

    cropped_img = img[y1_img_src:y2_img_src, x1_img_src:x2_img_src]

    pad_left, pad_top = -min(0, x1_crop), -min(0, y1_crop)

    padded_tile = np.full((tile_h, tile_w, 3), 114, dtype=np.uint8)
    padded_tile[pad_top:pad_top + cropped_img.shape[0], pad_left:pad_left + cropped_img.shape[1]] = cropped_img
    return padded_tile, x1_img_src, y1_img_src, x2_img_src, y2_img_src


def _place_tile(
    mosaic_img: np.ndarray,
    padded_tile: np.ndarray,
    tile_col: int,
    tile_row: int,
    tile_w: int,
    tile_h: int
) -> Tuple[int, int]:
    x_offset_mosaic, y_offset_mosaic = tile_col * tile_w, tile_row * tile_h
    mosaic_img[y_offset_mosaic:y_offset_mosaic + tile_h, x_offset_mosaic:x_offset_mosaic + tile_w] = padded_tile
    return x_offset_mosaic, y_offset_mosaic


def _project_annotations_to_tile(
    anns: np.ndarray,
    img_w: int,
    img_h: int,
    x1_crop: int,
    y1_crop: int,
    tile_w: int,
    tile_h: int,
    x_offset_mosaic: int,
    y_offset_mosaic: int,
    output_w: int,
    output_h: int
) -> List[List[float]]:
    final_tile_annotations = []
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
            final_tile_annotations.append([
                class_id,
                final_cx / output_w, final_cy / output_h,
                new_w / output_w, new_h / output_h
            ])
    return final_tile_annotations


def _apply_copy_paste(
    mosaic_img: np.ndarray,
    final_annotations: List[List[float]],
    extra_images: Optional[List[np.ndarray]],
    extra_annotations: Optional[List[np.ndarray]],
    output_w: int,
    output_h: int,
    copy_paste_prob: float
) -> None:
    # 5. Apply Copy-Paste Augmentation (Code remains the same)
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


# ---------------------------
# Public API: Mosaic
# ---------------------------

def mosaic(
    images: List[np.ndarray],
    annotations: List[np.ndarray],
    class_freqs: Dict[int, int],
    extra_images: Optional[List[np.ndarray]] = None,
    extra_annotations: Optional[List[np.ndarray]] = None,
    copy_paste_prob: float = 0.5,
    output_size: Tuple[int, int] = (640, 640),
    flip_prob: float = 0.2,
    affine_prob: float = 0.2,
    affine_degrees: float = 0.0,
    affine_scale: float = 0.1,
    affine_shear: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a class-balanced, object-centric mosaic with geometric and copy-paste augmentations.

    Args:
        images (List[np.ndarray]): List of all images.
        annotations (List[np.ndarray]): List of all annotation arrays.
        class_freqs (Dict[int, int]): Dictionary of class frequencies.
        extra_images (List[np.ndarray], optional): Images for copy-paste.
        extra_annotations (List[np.ndarray], optional): Annotations for extra images.
        copy_paste_prob (float): Probability of copy-paste augmentation.
        output_size (tuple): Final size of the mosaic image.
        flip_prob (float): Probability of horizontal flip.
        affine_prob (float): Probability of affine transformations.
        affine_degrees (float): Max rotation degrees for affine transform.
        affine_scale (float): Max scale factor for affine transform.
        affine_shear (float): Max shear degrees for affine transform.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The final mosaic image and its annotations.
    """
    n = _choose_mosaic_count()
    weights = _compute_sampling_weights(annotations, class_freqs)
    selected_images, selected_annotations, _ = _select_images_by_weight(images, annotations, n, weights)

    output_w, output_h = output_size
    grid_w, grid_h = _grid_layout(n)
    tile_w = output_w // grid_w
    tile_h = output_h // grid_h

    mosaic_img = np.full((output_h, output_w, 3), 114, dtype=np.uint8)
    final_annotations: List[List[float]] = []

    # 4. Build the mosaic with object-centric crops
    for i in range(n):
        img, anns = selected_images[i], selected_annotations[i]

        # --- APPLY NEW GEOMETRIC AUGMENTATIONS ---
        img, anns = random_flip(img.copy(), anns.copy(), prob=flip_prob)
        img, anns = random_affine(
            img, anns,
            degrees=affine_degrees,
            scale=affine_scale,
            shear=affine_shear,
            prob=affine_prob
        )
        # --- END OF NEW AUGMENTATIONS ---

        img_h, img_w, _ = img.shape
        center_x, center_y = _object_centric_center(img, anns)
        x1_crop, y1_crop = _compute_crop_coords(center_x, center_y, tile_w, tile_h)

        padded_tile, _, _, _, _ = _crop_and_pad_tile(img, x1_crop, y1_crop, tile_w, tile_h)

        tile_col, tile_row = i % grid_w, i // grid_w
        x_offset_mosaic, y_offset_mosaic = _place_tile(mosaic_img, padded_tile, tile_col, tile_row, tile_w, tile_h)

        tile_anns = _project_annotations_to_tile(
            anns, img_w, img_h, x1_crop, y1_crop, tile_w, tile_h,
            x_offset_mosaic, y_offset_mosaic, output_w, output_h
        )
        final_annotations.extend(tile_anns)

    _apply_copy_paste(
        mosaic_img=mosaic_img,
        final_annotations=final_annotations,
        extra_images=extra_images,
        extra_annotations=extra_annotations,
        output_w=output_w,
        output_h=output_h,
        copy_paste_prob=copy_paste_prob
    )

    return mosaic_img, np.array(final_annotations, dtype=np.float32)
