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
    gridmask_prob=0.3,
    grid_ratio=0.7,
    jitter_params=None,
    max_retry=10,
    seed=None
):
    """
    Builds an object-centric, class-balanced mosaic with photometric jitter, GridMask, and Copy-Paste.
    """

    # Seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Sample 4 images inversely by class frequency
    weights = []
    for anns in annotations:
        if len(anns):
            w = np.mean([1.0 / (class_freqs.get(int(a[0]), 1)) for a in anns])
        else:
            w = 1.0
        weights.append(w)
    weights = np.array(weights) / sum(weights)
    idxs = np.random.choice(len(images), size=4, replace=False, p=weights)

    # 2. Random tile size
    tile = random.choice(tile_sizes)
    w, h = tile, tile

    # 3. Prepare resized & jittered quadrants
    quad_imgs = []
    quad_anns = []
    for i, idx in enumerate(idxs):
        img = cv2.resize(images[idx], (w, h))
        anns = annotations[idx]

        count = len(anns)
        if jitter_params is None:
            mild = {'brightness': 0.2, 'noise_p': 0.3}
            strong = {'brightness': 0.5, 'noise_p': 0.3}
        else:
            mild = jitter_params['mild']
            strong = jitter_params['strong']

        p = mild if count >= 3 else strong

        # alpha = 1 + random.uniform(-p['brightness'], p['brightness'])
        # beta = random.uniform(-p['brightness']*50, p['brightness']*50)
        # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        #
        # if random.random() < p['noise_p']:
        #     noise = np.random.randn(h, w, 3) * 25
        #     img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        quad_imgs.append(img)
        quad_anns.append(anns)

    # 4. Create 2x2 mosaic canvas
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    offsets = [(0, 0), (w, 0), (0, h), (w, h)]
    for img_q, (xo, yo) in zip(quad_imgs, offsets):
        canvas[yo:yo+h, xo:xo+w] = img_q

    # 5. Object-centric crop around center with jitter
    for _ in range(max_retry):
        dx = int(random.uniform(-crop_offset, crop_offset) * w)
        dy = int(random.uniform(-crop_offset, crop_offset) * h)
        cx = w + dx
        cy = h + dy
        x0 = max(0, min(cx - w // 2, 2 * w - w))
        y0 = max(0, min(cy - h // 2, 2 * h - h))

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

        keep = []
        for (ax, ay, aw, ah), cls in zip(abs_boxes, abs_classes):
            if (x0 < ax < x0 + w) and (y0 < ay < y0 + h):
                keep.append((cls, ax, ay, aw, ah))
        if keep:
            break

    cropped = canvas[y0:y0+h, x0:x0+w]

    # 6. Update annotations to new crop frame
    new_anns = []
    for (cls, ax, ay, aw, ah) in keep:
        x_center = (ax - x0) / w
        y_center = (ay - y0) / h
        bw = aw / w
        bh = ah / h

        if 0 < x_center < 1 and 0 < y_center < 1:
            new_anns.append([cls, x_center, y_center, bw, bh])

    # 7. GridMask
    if random.random() < gridmask_prob and len(new_anns) > 0:
        mask = np.ones((h, w), dtype=np.uint8)
        d = int(w * grid_ratio)
        top = random.randint(0, d)
        left = random.randint(0, d)
        for y in range(-d, h, d * 2):
            for x in range(-d, w, d * 2):
                y1 = y + top
                x1 = x + left
                mask[max(0, y1):min(h, y1 + d), max(0, x1):min(w, x1 + d)] = 0
        cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

        # Remove annotations fully in black area
        visible_anns = []
        for ann in new_anns:
            cls, x_c, y_c, bw, bh = ann
            cx = int(x_c * w)
            cy = int(y_c * h)
            box_w = int(bw * w / 2)
            box_h = int(bh * h / 2)
            x0_box = max(0, cx - box_w)
            y0_box = max(0, cy - box_h)
            x1_box = min(w, cx + box_w)
            y1_box = min(h, cy + box_h)

            patch_mask = mask[y0_box:y1_box, x0_box:x1_box]
            visible_ratio = np.mean(patch_mask)  # 1 = fully visible, 0 = fully masked

            if visible_ratio > 0.3:  # keep box if 30%+ visible
                visible_anns.append(ann)

        new_anns = visible_anns

    # 8. Copy-Paste small objects
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

            # check bounds
            if x0o < 0 or y0o < 0 or x0o + w_obj > iw or y0o + h_obj > ih:
                pass  # skip invalid patch
            elif w_obj > 0 and h_obj > 0:
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
