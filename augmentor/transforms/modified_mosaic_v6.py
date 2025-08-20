import cv2
import numpy as np
import random


def random_flip(image, annotations, prob=0.5):
    """Applies a random horizontal flip to an image and its annotations."""
    if random.random() < prob:
        # Horizontally flip the image
        image = np.fliplr(image)
        # Adjust annotations if they exist
        if annotations is not None and len(annotations) > 0:
            # For YOLO format [class, cx, cy, w, h], only cx needs to be flipped
            annotations[:, 1] = 1.0 - annotations[:, 1]
    return image, annotations


def apply_copy_paste(
        image, annotations, extra_images, extra_annotations, prob=0.5, max_pastes=3
):
    """
    Applies copy-paste augmentation by pasting objects from extra images onto the main image.
    This function remains as you provided, as it correctly performs the copy-paste operation.
    """
    if random.random() > prob or not extra_images or not extra_annotations:
        return image, annotations

    output_h, output_w = image.shape[:2]
    num_pastes = random.randint(1, max_pastes)

    # Convert annotations to a list to easily append new ones
    new_annotations = list(annotations)

    for _ in range(num_pastes):
        # 1. Select a random source image and its annotations
        source_idx = random.randrange(len(extra_images))
        source_img = extra_images[source_idx]
        source_anns = extra_annotations[source_idx]

        if len(source_anns) == 0:
            continue

        # 2. Select a random object to copy from the source
        obj_ann = random.choice(source_anns)
        class_id, cx_n, cy_n, w_n, h_n = obj_ann

        # 3. Extract the object patch using its bounding box
        src_h, src_w = source_img.shape[:2]
        abs_w, abs_h = int(w_n * src_w), int(h_n * src_h)
        abs_x1 = int((cx_n * src_w) - (abs_w / 2))
        abs_y1 = int((cy_n * src_h) - (abs_h / 2))

        # Clamp coordinates to be within image bounds
        abs_x1, abs_y1 = max(0, abs_x1), max(0, abs_y1)
        patch = source_img[abs_y1:abs_y1 + abs_h, abs_x1:abs_x1 + abs_w]

        if patch.size == 0:
            continue

        # 4. Find a valid random location to paste the object
        patch_h, patch_w = patch.shape[:2]
        if patch_h >= output_h or patch_w >= output_w:
            continue

        paste_x = random.randint(0, output_w - patch_w)
        paste_y = random.randint(0, output_h - patch_h)

        # 5. Paste the object and add its annotation
        image[paste_y:paste_y + patch_h, paste_x:paste_x + patch_w] = patch

        new_cx = (paste_x + patch_w / 2) / output_w
        new_cy = (paste_y + patch_h / 2) / output_h
        new_w = patch_w / output_w
        new_h = patch_h / output_h

        new_annotations.append([class_id, new_cx, new_cy, new_w, new_h])

    return image, np.array(new_annotations)


def mosaic(
    images, annotations,
    extra_images=None, extra_annotations=None,
    output_size=(640, 640),
    flip_prob=0.5,
    crop_offset=0.15,
    copy_paste_prob=0.5,
    max_pastes=1
):
    """
    Creates a 4-image mosaic with random crop and applies copy-paste augmentation.

    Args:
        images (List[np.ndarray]): List of 4 base images.
        annotations (List[np.ndarray]): List of annotations for the 4 base images.
        extra_images (List[np.ndarray], optional): Pool of images for copy-paste.
        extra_annotations (List[np.ndarray], optional): Annotations for extra images.
        output_size (tuple): Final size of the mosaic image (width, height).
        flip_prob (float): Probability of horizontal flip for each base image.
        crop_offset (float): Maximum random crop offset as a fraction of image size.
        copy_paste_prob (float): Probability of applying the copy-paste augmentation.
        max_pastes (int): Maximum number of objects to paste.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The final augmented image and its annotations.
    """
    assert len(images) == 4, "Exactly four images are required for mosaic."
    assert len(annotations) == 4, "Exactly four annotation sets are required."

    w, h = output_size

    # --- Part 1: Create the Base Mosaic ---

    # Independently flip each of the 4 images
    processed_images = []
    processed_annotations = []
    for img, anns in zip(images, annotations):
        img_flipped, anns_flipped = random_flip(img.copy(), anns.copy(), prob=flip_prob)
        processed_images.append(img_flipped)
        processed_annotations.append(anns_flipped)

    # Resize images for the mosaic canvas
    resized_images = [cv2.resize(img, output_size) for img in processed_images]

    # Create the 2x2 mosaic canvas
    mosaic_canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    quadrant_offsets = [(0, 0), (w, 0), (0, h), (w, h)]

    for i, (x_off, y_off) in enumerate(quadrant_offsets):
        mosaic_canvas[y_off:y_off + h, x_off:x_off + w] = resized_images[i]

    # Apply a random crop offset
    x_offset_crop = int(random.uniform(-crop_offset, crop_offset) * w)
    y_offset_crop = int(random.uniform(-crop_offset, crop_offset) * h)
    x_center = w + x_offset_crop
    y_center = h + y_offset_crop

    crop_x1 = x_center - w // 2
    crop_y1 = y_center - h // 2
    cropped_mosaic = mosaic_canvas[crop_y1:crop_y1 + h, crop_x1:crop_x1 + w]

    # Calculate annotations for the newly formed mosaic
    mosaic_annotations = []
    for i, anns in enumerate(processed_annotations):
        quad_x_off, quad_y_off = quadrant_offsets[i]
        for ann in anns:
            class_id, x_norm, y_norm, bw_norm, bh_norm = ann

            # Absolute coordinates on the 2x2 canvas
            abs_x = x_norm * w + quad_x_off
            abs_y = y_norm * h + quad_y_off
            abs_bw = bw_norm * w
            abs_bh = bh_norm * h

            # Corners on the 2x2 canvas
            x1_canvas, y1_canvas = abs_x - abs_bw / 2, abs_y - abs_bh / 2
            x2_canvas, y2_canvas = abs_x + abs_bw / 2, abs_y + abs_bh / 2

            # Corners relative to the final cropped image
            x1_final = x1_canvas - crop_x1
            y1_final = y1_canvas - crop_y1
            x2_final = x2_canvas - crop_x1
            y2_final = y2_canvas - crop_y1

            # Clip bounding box to the final image boundaries
            x1_final = max(0, min(w, x1_final))
            y1_final = max(0, min(h, y1_final))
            x2_final = max(0, min(w, x2_final))
            y2_final = max(0, min(h, y2_final))

            # Skip if the box is no longer valid
            if x2_final <= x1_final or y2_final <= y1_final:
                continue

            # Convert back to normalized YOLO format [cx, cy, w, h]
            new_w = (x2_final - x1_final)
            new_h = (y2_final - y1_final)
            new_center_x = (x1_final + new_w / 2)
            new_center_y = (y1_final + new_h / 2)

            mosaic_annotations.append([
                int(class_id), new_center_x / w, new_center_y / h, new_w / w, new_h / h
            ])

    mosaic_annotations = np.array(mosaic_annotations)

    # --- Part 2: Apply Copy-Paste on the finished mosaic ---

    final_image, final_annotations = apply_copy_paste(
        cropped_mosaic.copy(),  # Use a copy to avoid in-place modification issues
        mosaic_annotations,
        extra_images,
        extra_annotations,
        prob=copy_paste_prob,
        max_pastes=max_pastes
    )

    return final_image, final_annotations
