import cv2
import numpy as np
import random


def enhanced_mosaic(
    images,
    annotations,
    class_freqs,
    extra_images=None,
    extra_annotations=None,
    tile_sizes=(512, 640, 768),
    crop_offset=0.2,
    copy_paste_prob=0.4,
    gridmask_prob=0.3,
    grid_ratio=0.7,
    jitter_params=None,
    max_retry=10
):
    """
    Builds an object-centric, class-balanced mosaic with photometric jitter, GridMask, and Copy-Paste.

    Args:
        images (List[np.ndarray]): List of input images.
        annotations (List[np.ndarray]): Corresponding YOLO-format annotations per image.
        class_freqs (Dict[int, int]): Mapping class_id -> frequency for sampling.
        extra_images (List[np.ndarray], optional): Pool for copy-paste source images.
        extra_annotations (List[np.ndarray], optional): Corresponding annotations.
        tile_sizes (Tuple[int]): Possible tile edge sizes.
        crop_offset (float): Max jitter fraction around center.
        copy_paste_prob (float): Probability of doing copy-paste.
        gridmask_prob (float): Probability of applying GridMask on final.
        grid_ratio (float): Fraction of masked area per GridMask cell.
        jitter_params (dict, optional): Per-quadrant jitter config.
        max_retry (int): Max attempts for object-centric crop.

    Returns:
        mosaic_img, updated_anns
    """
    # 1. Sample 4 images inversely by class frequency
    weights = []
    for anns in annotations:
        # weight = sum(1/freq[class] for each object), avg
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

        # count objects -> choose jitter strength
        count = len(anns)
        if jitter_params is None:
            # defaults
            mild = {'brightness': 0.2, 'noise_p': 0.3}
            strong = {'brightness': 0.5, 'noise_p': 0.3}
        else:
            mild = jitter_params['mild']
            strong = jitter_params['strong']

        p = mild if count >= 3 else strong
        # brightness/contrast
        alpha = 1 + random.uniform(-p['brightness'], p['brightness'])
        beta = random.uniform(-p['brightness']*50, p['brightness']*50)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # noise
        if random.random() < p['noise_p']:
            noise = np.random.randn(h, w, 3) * 25
            img = cv2.add(img, noise.astype(np.uint8))

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
        x0 = cx - w // 2
        y0 = cy - h // 2
        # collect absolute boxes
        abs_boxes = []
        for (anns, (xo, yo)) in zip(quad_anns, offsets):
            for a in anns:
                _, x_c, y_c, bw, bh = a
                ax = int(x_c * w) + xo
                ay = int(y_c * h) + yo
                abs_w = bw * w
                abs_h = bh * h
                abs_boxes.append((ax, ay, abs_w, abs_h))
        # check if any remain in crop
        keep = []
        for box in abs_boxes:
            ax, ay, aw, ah = box
            if (ax >= x0 and ax <= x0 + w and ay >= y0 and ay <= y0 + h):
                keep.append(box)
        if keep:
            break
    cropped = canvas[y0:y0+h, x0:x0+w]

    # 6. Update annotations to new crop frame
    new_anns = []
    for (anns, (xo, yo)) in zip(quad_anns, offsets):
        for a in anns:
            cls, x_c, y_c, bw, bh = a
            ax = x_c * w + xo
            ay = y_c * h + yo
            nx = (ax - x0) / w
            ny = (ay - y0) / h
            # clip
            if nx<0 or nx>1 or ny<0 or ny>1:
                continue
            # size remain same bw,bh
            new_anns.append([int(cls), nx, ny, bw, bh])

    # 7. GridMask
    if random.random() < gridmask_prob:
        mask = np.ones((h, w), dtype=np.uint8)
        d = int(w * grid_ratio)
        top = random.randint(0, d)
        left = random.randint(0, d)
        for y in range(-d, h, d*2):
            for x in range(-d, w, d*2):
                y1 = y + top
                x1 = x + left
                mask[max(0,y1):min(h,y1+d), max(0,x1):min(w,x1+d)] = 0
        cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

    # 8. Copy-Paste small objects
    if extra_images and extra_annotations and random.random() < copy_paste_prob:
        # pick one random small object
        ei = random.randrange(len(extra_images))
        eanns = extra_annotations[ei]
        if len(eanns):
            a = random.choice(eanns)
            cls, x_c, y_c, bw, bh = a
            src = extra_images[ei]
            ih, iw = src.shape[:2]
            # crop object patch
            w_obj = int(bw * iw)
            h_obj = int(bh * ih)
            x0o = int((x_c - bw/2) * iw)
            y0o = int((y_c - bh/2) * ih)
            patch = src[y0o:y0o+h_obj, x0o:x0o+w_obj]
            # paste at random location
            px = random.randint(0, w-w_obj)
            py = random.randint(0, h-h_obj)
            cropped[py:py+h_obj, px:px+w_obj] = patch
            new_anns.append([int(cls), (px + w_obj/2)/w, (py + h_obj/2)/h, bw, bh])

    return cropped, np.array(new_anns)
