from pathlib import Path

import cv2
import numpy as np
import random


def enhanced_mosaic_all_with_augmentations(images, annotations, class_freqs,
                                           output_size=(640, 640), crop_offset=0.15,
                                           flip_prob=0.5, blur_prob=0.3,
                                           minority_threshold=0.3, max_paste_per_img=2):
    """
    Enhanced mosaic that guarantees 100% mosaic generation with blur and copy-paste.

    Args:
        images: List of 4 input images
        annotations: List of 4 annotation arrays
        class_freqs: Dictionary of class frequencies
        output_size: Output dimensions (width, height)
        crop_offset: Random crop offset for mosaic center
        flip_prob: Probability of horizontal flip per tile
        blur_prob: Probability of applying Gaussian blur to final mosaic
        minority_threshold: Threshold for identifying minority classes
        max_paste_per_img: Max objects to paste onto mosaic

    Returns:
        Tuple of (mosaic_image, final_annotations)
    """
    assert len(images) == 4, "Exactly four images are required for mosaic."
    assert len(annotations) == 4, "Exactly four annotation sets are required."

    w, h = output_size
    tile_w, tile_h = w, h

    # Create 2x2 mosaic canvas
    mosaic_canvas = np.full((h * 2, w * 2, 3), 114, dtype=np.uint8)
    quadrant_offsets = [(0, 0), (w, 0), (0, h), (w, h)]

    updated_annotations = []

    # Process each tile with object-centric cropping and optional flip
    for i, (img, ann_list) in enumerate(zip(images, annotations)):
        img_h, img_w = img.shape[:2]
        quad_x_off, quad_y_off = quadrant_offsets[i]

        # Apply horizontal flip augmentation only
        ann_array = np.array(ann_list) if len(ann_list) > 0 else np.array([])
        if random.random() < flip_prob:
            img = np.fliplr(img)
            if len(ann_array) > 0:
                ann_array[:, 1] = 1.0 - ann_array[:, 1]  # flip cx coordinate

        # Object-centric cropping: choose random annotation center
        if len(ann_array) > 0:
            center_ann = random.choice(ann_array)
            _, cx, cy, _, _ = center_ann
            center_x, center_y = int(cx * img_w), int(cy * img_h)
        else:
            center_x, center_y = img_w // 2, img_h // 2

        # Calculate crop window centered on chosen point
        x1_crop = max(center_x - tile_w // 2, 0)
        y1_crop = max(center_y - tile_h // 2, 0)
        x2_crop = min(x1_crop + tile_w, img_w)
        y2_crop = min(y1_crop + tile_h, img_h)

        # Ensure we don't exceed bounds
        if x2_crop - x1_crop < tile_w:
            x1_crop = max(0, x2_crop - tile_w)
        if y2_crop - y1_crop < tile_h:
            y1_crop = max(0, y2_crop - tile_h)

        # Crop and pad tile
        cropped_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
        padded_tile = np.full((tile_h, tile_w, 3), 114, dtype=np.uint8)
        crop_h, crop_w = cropped_img.shape[:2]
        padded_tile[:crop_h, :crop_w] = cropped_img

        # Place tile in mosaic canvas
        mosaic_canvas[quad_y_off:quad_y_off + tile_h,
        quad_x_off:quad_x_off + tile_w] = padded_tile

        # Update annotations for this tile
        for ann in ann_array:
            class_id, cx_n, cy_n, w_n, h_n = ann

            # Convert to absolute coordinates
            abs_cx, abs_cy = cx_n * img_w, cy_n * img_h
            abs_w, abs_h = w_n * img_w, h_n * img_h
            abs_x1, abs_y1 = abs_cx - abs_w / 2, abs_cy - abs_h / 2

            # Calculate coordinates relative to cropped tile
            new_x1 = max(abs_x1 - x1_crop, 0)
            new_y1 = max(abs_y1 - y1_crop, 0)
            new_x2 = min(abs_x1 + abs_w - x1_crop, tile_w)
            new_y2 = min(abs_y1 + abs_h - y1_crop, tile_h)

            new_w = new_x2 - new_x1
            new_h = new_y2 - new_y1

            if new_w > 1 and new_h > 1:
                # Convert to mosaic canvas coordinates
                final_cx = (new_x1 + new_w / 2 + quad_x_off) / (2 * w)
                final_cy = (new_y1 + new_h / 2 + quad_y_off) / (2 * h)
                final_w = new_w / (2 * w)
                final_h = new_h / (2 * h)

                updated_annotations.append([int(class_id), final_cx, final_cy, final_w, final_h])

    # Apply random crop offset to final mosaic
    x_offset_crop = int(random.uniform(-crop_offset, crop_offset) * w)
    y_offset_crop = int(random.uniform(-crop_offset, crop_offset) * h)
    x_center = w + x_offset_crop
    y_center = h + y_offset_crop
    crop_x = max(0, min(w, x_center - w // 2))
    crop_y = max(0, min(h, y_center - h // 2))

    # Final crop to output size
    cropped_mosaic = mosaic_canvas[crop_y:crop_y + h, crop_x:crop_x + w]

    # Adjust annotations for final crop
    final_annotations = []
    for ann in updated_annotations:
        class_id, cx, cy, bw, bh = ann

        abs_cx = cx * (2 * w) - crop_x
        abs_cy = cy * (2 * h) - crop_y

        if 0 <= abs_cx <= w and 0 <= abs_cy <= h:
            x1 = max(0, abs_cx - bw * (2 * w) / 2)
            y1 = max(0, abs_cy - bh * (2 * h) / 2)
            x2 = min(w, abs_cx + bw * (2 * w) / 2)
            y2 = min(h, abs_cy + bh * (2 * h) / 2)

            new_w = x2 - x1
            new_h = y2 - y1

            if new_w > 1 and new_h > 1:
                new_cx = (x1 + new_w / 2) / w
                new_cy = (y1 + new_h / 2) / h
                new_bw = new_w / w
                new_bh = new_h / h

                final_annotations.append([class_id, new_cx, new_cy, new_bw, new_bh])

    # Apply Gaussian blur to the complete mosaic
    if random.random() < blur_prob:
        sigma = random.uniform(0.1, 1.0)
        kernel_size = int(sigma * 6) | 1  # Ensure odd kernel size
        cropped_mosaic = cv2.GaussianBlur(cropped_mosaic, (kernel_size, kernel_size), sigma)

    # Apply targeted copy-paste for minority classes
    final_img, final_anns = apply_targeted_copy_paste(
        cropped_mosaic, np.array(final_annotations),
        images, annotations, class_freqs,
        minority_threshold, max_paste_per_img
    )

    return final_img, final_anns


def apply_targeted_copy_paste(mosaic_img, mosaic_anns, source_images, source_annotations,
                              class_freqs, minority_threshold=0.3, max_paste=2):
    """
    Apply targeted copy-paste to mosaic for minority class balancing.

    Args:
        mosaic_img: The mosaic image to paste objects onto
        mosaic_anns: Current mosaic annotations
        source_images: Original source images for extracting objects
        source_annotations: Annotations for source images
        class_freqs: Dictionary of class frequencies
        minority_threshold: Threshold for identifying minority classes
        max_paste: Maximum objects to paste

    Returns:
        Tuple of (augmented_image, augmented_annotations)
    """
    total_instances = sum(class_freqs.values())
    minority_classes = [cls for cls, freq in class_freqs.items()
                        if freq / total_instances < minority_threshold]

    if not minority_classes:
        return mosaic_img, mosaic_anns

    # Collect minority class objects from source images
    source_objects = []
    for src_img, src_anns in zip(source_images, source_annotations):
        for ann in src_anns:
            class_id = int(ann[0])
            if class_id in minority_classes:
                source_objects.append((src_img, ann))

    if not source_objects:
        return mosaic_img, mosaic_anns

    # Select random objects to paste
    num_to_paste = min(max_paste, len(source_objects))
    selected_objects = random.sample(source_objects, num_to_paste)

    # Create augmented image and annotations
    aug_img = mosaic_img.copy()
    aug_anns = mosaic_anns.tolist()
    target_h, target_w = aug_img.shape[:2]

    for src_img, obj_ann in selected_objects:
        cls_id, cx, cy, w, h = obj_ann
        src_h, src_w = src_img.shape[:2]

        # Extract object patch
        abs_w, abs_h = int(w * src_w), int(h * src_h)
        abs_x1 = max(0, int((cx * src_w) - abs_w / 2))
        abs_y1 = max(0, int((cy * src_h) - abs_h / 2))
        abs_x2 = min(src_w, abs_x1 + abs_w)
        abs_y2 = min(src_h, abs_y1 + abs_h)

        if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
            continue

        patch = src_img[abs_y1:abs_y2, abs_x1:abs_x2]
        if patch.size == 0:
            continue

        # Find valid paste location
        patch_h, patch_w = patch.shape[:2]
        if patch_h >= target_h or patch_w >= target_w:
            continue

        paste_x = random.randint(0, target_w - patch_w)
        paste_y = random.randint(0, target_h - patch_h)

        # Paste object
        aug_img[paste_y:paste_y + patch_h, paste_x:paste_x + patch_w] = patch

        # Add new annotation
        new_cx = (paste_x + patch_w / 2) / target_w
        new_cy = (paste_y + patch_h / 2) / target_h
        new_w = patch_w / target_w
        new_h = patch_h / target_h

        aug_anns.append([cls_id, new_cx, new_cy, new_w, new_h])

    return aug_img, np.array(aug_anns)


def create_augmented_dataset(images, annotations_list, class_freqs, output_dir, target_count):
    """
    Generate exactly 'target_count' mosaic images by cycling through the original dataset.
    Each mosaic combines 4 images with blur and targeted copy-paste augmentation.

    Args:
        images: List of original training images
        annotations_list: List of annotations for each image
        class_freqs: Dictionary of class frequencies for minority class identification
        output_dir: Directory to save augmented images and labels
        target_count: Number of mosaic images to generate (e.g., 5000)

    Returns:
        Number of mosaic images generated
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    num_original_images = len(images)

    for i in range(target_count):
        # Use modulo arithmetic to cycle through dataset
        base_idx = (i * 4) % num_original_images

        # Select 4 images for this mosaic (with wraparound)
        group_images = [images[(base_idx + j) % num_original_images] for j in range(4)]
        group_annotations = [annotations_list[(base_idx + j) % num_original_images] for j in range(4)]

        # Generate mosaic with all augmentations
        mosaic_img, mosaic_anns = enhanced_mosaic_all_with_augmentations(
            group_images, group_annotations, class_freqs,
            flip_prob=0.5, blur_prob=0.3, minority_threshold=0.3, max_paste_per_img=2
        )

        # Save mosaic image
        img_filename = f"mosaic_{i:06d}.jpg"
        img_path = images_dir / img_filename
        cv2.imwrite(str(img_path), mosaic_img)

        # Save annotations
        label_filename = f"mosaic_{i:06d}.txt"
        label_path = labels_dir / label_filename
        with open(label_path, 'w') as f:
            for ann in mosaic_anns:
                f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

    print(f"Successfully generated {target_count} mosaic images")
    return target_count

# Usage:
# create_offline_mosaic_dataset(images, annotations_list, class_freqs, "mosaic_dataset")
