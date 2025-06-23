import cv2
import numpy as np
import random

def random_brightness_contrast(image, p=0.5):
    if random.random() > p:
        return image
    brightness = random.uniform(-0.2, 0.2)
    contrast = random.uniform(-0.2, 0.2)
    beta = brightness * 255
    alpha = 1.0 + contrast
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def random_gaussian_noise(image, p=0.3, mean=0.0, std=20.0):
    if random.random() > p:
        return image
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def mosaic(images, annotations, output_size=(640, 640), crop_offset=0.15):
    assert len(images) == 4, "Exactly four images_o are required for mosaic."
    assert len(annotations) == 4, "Exactly four annotation sets are required."

    w, h = output_size
    # Apply random augmentations per image BEFORE resizing
    processed_images = []
    for img in images:
        img_aug = random_brightness_contrast(img, p=0.5)
        img_aug = random_gaussian_noise(img_aug, p=0.3)
        processed_images.append(cv2.resize(img_aug, output_size))

    mosaic_canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    quadrant_offsets = [(0, 0), (w, 0), (0, h), (w, h)]

    for i, (x_off, y_off) in enumerate(quadrant_offsets):
        mosaic_canvas[y_off:y_off + h, x_off:x_off + w] = processed_images[i]

    x_offset_crop = int(random.uniform(-crop_offset, crop_offset) * w)
    y_offset_crop = int(random.uniform(-crop_offset, crop_offset) * h)
    x_center = w + x_offset_crop
    y_center = h + y_offset_crop
    crop_x = x_center - w // 2
    crop_y = y_center - h // 2
    cropped_mosaic = mosaic_canvas[crop_y:crop_y + h, crop_x:crop_x + w]

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

    return cropped_mosaic, np.array(updated_annotations)
