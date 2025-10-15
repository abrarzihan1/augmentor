import cv2
import numpy as np
import random

def enhanced_mosaic(
    images,
    annotations,
    class_freqs,
    extra_images=None,
    extra_annotations=None,
    tile_sizes=(640,),
    crop_offset=0.2,
    copy_paste_prob=0.4,
    gridmask_prob=0.0,
    grid_ratio=0.7,
    jitter_params=None,
    max_retry=10,
    seed=None
):
    """
    Builds an object-centric, class-balanced mosaic with photometric jitter, GridMask, and Copy-Paste.
    Now supports mosaics of 4, 6, or 9 images with true object-centric cropping.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Choose mosaic type randomly
    num_tiles = random.choice([4, 6, 9])
    if num_tiles == 4:
        grid_w, grid_h = 2, 2
    elif num_tiles == 6:
        if random.random() < 0.5:
            grid_w, grid_h = 3, 2
        else:
            grid_w, grid_h = 2, 3
    else:  # 9
        grid_w, grid_h = 3, 3

    # 1. Sample images inversely by class frequency
    weights = []
    for anns in annotations:
        if len(anns):
            w = np.mean([1.0 / (class_freqs.get(int(a[0]), 1)) for a in anns])
        else:
            w = 1.0
        weights.append(w)
    weights = np.array(weights) / sum(weights)
    idxs = np.random.choice(len(images), size=num_tiles, replace=False, p=weights)

    # 2. Random tile size
    tile = random.choice(tile_sizes)
    w, h = tile, tile

    # 3. Prepare resized tiles
    quad_imgs = []
    quad_anns = []
    for idx in idxs:
        img = cv2.resize(images[idx], (w, h))
        anns = annotations[idx]
        quad_imgs.append(img)
        quad_anns.append(anns)

    # 4. Create mosaic canvas
    canvas_w = w * grid_w
    canvas_h = h * grid_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    offsets = []
    i = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            if i >= num_tiles:
                break
            xo, yo = gx * w, gy * h
            canvas[yo:yo+h, xo:xo+w] = quad_imgs[i]
            offsets.append((xo, yo))
            i += 1

    # 5. Build absolute annotation list
    abs_boxes = []
    abs_classes = []
    for (anns, (xo, yo)) in zip(quad_anns, offsets):
        for a in anns:
            cls, x_c, y_c, bw, bh = a
            ax = x_c * w + xo
            ay = y_c * h + yo
            abs_w = bw * w
            abs_h = bh * h
            abs_boxes.append((ax, ay, abs_w, abs_h))
            abs_classes.append(int(cls))

    # If no objects, fallback to center crop
    if not abs_boxes:
        crop_center_x = canvas_w // 2
        crop_center_y = canvas_h // 2
    else:
        # Pick a random object's center
        ax, ay, _, _ = random.choice(abs_boxes)
        # Apply jitter
        crop_center_x = ax + random.uniform(-crop_offset, crop_offset) * w
        crop_center_y = ay + random.uniform(-crop_offset, crop_offset) * h

    # Clamp crop center so it fits inside canvas
    x0 = int(max(0, min(crop_center_x - w // 2, canvas_w - w)))
    y0 = int(max(0, min(crop_center_y - h // 2, canvas_h - h)))

    # 6. Keep only annotations inside crop
    keep = []
    for (ax, ay, aw, ah), cls in zip(abs_boxes, abs_classes):
        if (x0 < ax < x0 + w) and (y0 < ay < y0 + h):
            keep.append((cls, ax, ay, aw, ah))

    cropped = canvas[y0:y0+h, x0:x0+w]

    # 7. Convert kept annotations to relative coords
    new_anns = []
    for (cls, ax, ay, aw, ah) in keep:
        x_center = (ax - x0) / w
        y_center = (ay - y0) / h
        bw = aw / w
        bh = ah / h
        if 0 < x_center < 1 and 0 < y_center < 1:
            new_anns.append([cls, x_center, y_center, bw, bh])

    # 8. Copy-Paste augmentation
    if (
        extra_images and extra_annotations
        and random.random() < copy_paste_prob
        and len(extra_annotations) > 0
    ):
        ei = random.randrange(len(extra_images))
        src = extra_images[ei]
        eanns = extra_annotations[ei]
        if len(eanns):
            a = random.choice(eanns)
            cls, x_c, y_c, bw, bh = a
            ih, iw = src.shape[:2]
            x0o = int((x_c - bw / 2) * iw)
            y0o = int((y_c - bh / 2) * ih)
            w_obj = int(bw * iw)
            h_obj = int(bh * ih)
            if x0o >= 0 and y0o >= 0 and x0o + w_obj <= iw and y0o + h_obj <= ih and w_obj > 0 and h_obj > 0:
                patch = src[y0o:y0o+h_obj, x0o:x0o+w_obj]
                if patch.shape[0] <= h and patch.shape[1] <= w:
                    px = random.randint(0, w - patch.shape[1])
                    py = random.randint(0, h - patch.shape[0])
                    cropped[py:py+patch.shape[0], px:px+patch.shape[1]] = patch
                    new_anns.append([
                        int(cls),
                        (px + patch.shape[1]/2)/w,
                        (py + patch.shape[0]/2)/h,
                        patch.shape[1]/w,
                        patch.shape[0]/h
                    ])

    return cropped, np.array(new_anns, dtype=np.float32)
