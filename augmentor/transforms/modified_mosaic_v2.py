"""
Enhanced mosaic generator for YOLO-style datasets.

Features included:
- Mild per-tile HSV/gamma jitter
- Tile scale jitter and random overlap when composing the 2x2 mosaic
- Seam feathering to smooth tile borders
- Global photometric harmonization after cropping
- Copy-paste with rotation/scale and feathered alpha + optional LAB color matching
- Optional GridMask or Cutout
- Object-centric cropping with fallback to random crop

Inputs expected:
- images: list of HxWx3 uint8 numpy arrays (BGR as used by OpenCV)
- annotations: list of arrays/lists of annotations in YOLO format [class, x_center_rel, y_center_rel, w_rel, h_rel]
- class_freqs: dict mapping class_id -> frequency (for inverse sampling weights)

Returns a tuple (cropped_image, new_annotations_array)

Note: This is a standalone module for offline augmentation. Integrate carefully and validate bbox math when you change tile sizing strategies.
"""

from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
import random


# --------------------------- Photometric utilities ---------------------------

def random_hsv(img: np.ndarray, hue_delta=8, sat_scale=0.12, val_scale=0.12) -> np.ndarray:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = img_hsv[..., 0]
    s = img_hsv[..., 1]
    v = img_hsv[..., 2]

    dh = random.uniform(-hue_delta, hue_delta)
    ds = 1.0 + random.uniform(-sat_scale, sat_scale)
    dv = 1.0 + random.uniform(-val_scale, val_scale)

    h = (h + dh) % 180.0
    s = np.clip(s * ds, 0, 255)
    v = np.clip(v * dv, 0, 255)

    img_hsv[..., 0] = h
    img_hsv[..., 1] = s
    img_hsv[..., 2] = v
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_gamma(img: np.ndarray, gamma_range=(0.85, 1.25)) -> np.ndarray:
    g = random.uniform(*gamma_range)
    inv = 1.0 / g
    table = (np.arange(256) / 255.0) ** inv * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


def maybe_blur_or_sharpen(img: np.ndarray, p_blur=0.12, p_sharp=0.05) -> np.ndarray:
    r = random.random()
    if r < p_blur:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    elif r < p_blur + p_sharp:
        blur = cv2.GaussianBlur(img, (3, 3), 1)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img


def add_noise(img: np.ndarray, noise_p=0.06) -> np.ndarray:
    if random.random() >= noise_p:
        return img
    r = random.random()
    out = img.astype(np.float32)
    if r < 0.6:
        sigma = random.uniform(3, 14)
        noise = np.random.randn(*img.shape) * sigma
        out = out + noise
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        # mild salt and pepper
        prob = random.uniform(0.0005, 0.006)
        mask = np.random.choice([0, 1, 2], size=img.shape[:2], p=[1 - prob, prob / 2, prob / 2])
        out = img.copy()
        out[mask == 1] = 0
        out[mask == 2] = 255
        return out


def global_photometric(img: np.ndarray, prob=1.0) -> np.ndarray:
    if random.random() > prob:
        return img
    out = img.copy()
    out = random_hsv(out, hue_delta=6, sat_scale=0.10, val_scale=0.10)
    out = random_gamma(out, gamma_range=(0.9, 1.15))
    out = maybe_blur_or_sharpen(out, p_blur=0.10, p_sharp=0.04)
    out = add_noise(out, noise_p=0.04)
    if random.random() < 0.08:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


# --------------------------- Copy-paste utilities ---------------------------

def paste_with_feather(dst: np.ndarray, patch: np.ndarray, px: int, py: int, feather: int = 8,
                       color_match: bool = True) -> np.ndarray:
    h, w = patch.shape[:2]
    H, W = dst.shape[:2]
    if h == 0 or w == 0:
        return dst
    if px < 0 or py < 0 or px + w > W or py + h > H:
        return dst  # skip out-of-bounds

    # color match in LAB space (match mean and std)
    dst_region = dst[py:py + h, px:px + w]
    patch_f = patch.astype(np.float32)
    dst_f = dst_region.astype(np.float32)

    if color_match:
        patch_lab = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        dst_lab = cv2.cvtColor(dst_region.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        for c in range(3):
            pm = patch_lab[..., c].mean()
            ps = patch_lab[..., c].std() + 1e-6
            dm = dst_lab[..., c].mean()
            ds = dst_lab[..., c].std() + 1e-6
            patch_lab[..., c] = (patch_lab[..., c] - pm) * (ds / ps) + dm
        patch_f = cv2.cvtColor(np.clip(patch_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    # create alpha mask with feather edges
    alpha = np.ones((h, w), dtype=np.float32)
    if feather > 0:
        ys, xs = np.ogrid[:h, :w]
        dist = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys)).astype(np.float32)
        alpha = np.clip(dist / float(feather), 0.0, 1.0)

    alpha = alpha[..., None]
    blended = (patch_f * alpha + dst_f * (1.0 - alpha)).astype(np.uint8)
    out = dst.copy()
    out[py:py + h, px:px + w] = blended
    return out


# --------------------------- Seam blending ---------------------------

def blend_seams(canvas: np.ndarray, tile_w: int, tile_h: int, seam_width: int = 12) -> np.ndarray:
    H, W = canvas.shape[:2]
    out = canvas.copy().astype(np.float32)
    # vertical seam at x = tile_w
    for dx in range(-seam_width, seam_width):
        alpha = (dx + seam_width) / (2.0 * seam_width)
        x = tile_w + dx
        if x <= 0 or x >= W - 1:
            continue
        left = canvas[:, x - 1, :].astype(np.float32)
        right = canvas[:, x, :].astype(np.float32)
        out[:, x, :] = left * (1 - alpha) + right * alpha
    # horizontal seam at y = tile_h
    for dy in range(-seam_width, seam_width):
        alpha = (dy + seam_width) / (2.0 * seam_width)
        y = tile_h + dy
        if y <= 0 or y >= H - 1:
            continue
        top = canvas[y - 1, :, :].astype(np.float32)
        bottom = canvas[y, :, :].astype(np.float32)
        out[y, :, :] = top * (1 - alpha) + bottom * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------- GridMask / Cutout ---------------------------

def apply_gridmask(img: np.ndarray, grid_ratio=0.7) -> np.ndarray:
    h, w = img.shape[:2]
    d = int(max(8, w * grid_ratio))
    mask = np.ones((h, w), dtype=np.uint8)
    top = random.randint(0, max(0, d - 1))
    left = random.randint(0, max(0, d - 1))
    for y in range(-d, h, d * 2):
        for x in range(-d, w, d * 2):
            y1 = y + top
            x1 = x + left
            mask[max(0, y1):min(h, y1 + d), max(0, x1):min(w, x1 + d)] = 0
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


# --------------------------- Main enhanced mosaic ---------------------------

def enhanced_mosaic_v2(
    images: List[np.ndarray],
    annotations: List[np.ndarray],
    class_freqs: Dict[int, float],
    extra_images: Optional[List[np.ndarray]] = None,
    extra_annotations: Optional[List[np.ndarray]] = None,
    tile_size: int = 640,
    crop_offset: float = 0.15,
    copy_paste_prob: float = 0.45,
    gridmask_prob: float = 0.25,
    grid_ratio: float = 0.6,
    max_retry: int = 8,
    seed: Optional[int] = None,
    allow_overlap: bool = True,
    seam_width: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one enhanced mosaic image + annotations.

    Returns: cropped_img (tile_size x tile_size x 3), new_anns (N x 5) in YOLO format
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N_images = len(images)
    if N_images == 0:
        raise ValueError("no images provided")

    # 1. Sampling weights (inverse class freq if objects exist)
    weights = []
    for anns in annotations:
        if len(anns):
            w = np.mean([1.0 / (class_freqs.get(int(a[0]), 1.0)) for a in anns])
        else:
            w = 1.0
        weights.append(w)
    weights = np.array(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    # choose 4 indices without replacement (if less than 4 images, allow replacement)
    replace = (N_images < 4)
    idxs = np.random.choice(N_images, size=4, replace=replace, p=weights)

    # 2. prepare a canvas (2x2 tiles)
    tile_w, tile_h = tile_size, tile_size
    canvas_h = tile_h * 2
    canvas_w = tile_w * 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    offsets = []
    quad_abs_ann = []  # absolute bbox entries: (cls, cx_abs, cy_abs, w_abs, h_abs)

    # optional random overlap offsets per quadrant
    for i, idx in enumerate(idxs):
        src = images[idx]
        anns = annotations[idx]
        ih, iw = src.shape[:2]

        # scale jitter (random scale) and optional rotation
        scale = random.uniform(0.7, 1.3)
        new_w = max(2, int(iw * scale))
        new_h = max(2, int(ih * scale))
        src_scaled = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # choose a crop region from scaled image to fill tile (allow partial)
        if new_w > tile_w:
            xoff = random.randint(0, new_w - tile_w)
        else:
            xoff = 0
        if new_h > tile_h:
            yoff = random.randint(0, new_h - tile_h)
        else:
            yoff = 0

        tile_img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tile_img[:min(tile_h, new_h - yoff), :min(tile_w, new_w - xoff)] = \
            src_scaled[yoff:yoff + min(tile_h, new_h - yoff), xoff:xoff + min(tile_w, new_w - xoff)]

        # per-tile mild photometric jitter
        if random.random() < 0.9:
            tile_img = random_hsv(tile_img, hue_delta=6, sat_scale=0.08, val_scale=0.08)
        if random.random() < 0.25:
            tile_img = random_gamma(tile_img, gamma_range=(0.95, 1.08))
        if random.random() < 0.08:
            tile_img = add_noise(tile_img, noise_p=0.5)

        # compute where to place this tile in canvas
        base_x = (i % 2) * tile_w
        base_y = (i // 2) * tile_h
        if allow_overlap:
            # small random offsets to allow overlap/cross-tile objects
            ox = int(random.uniform(-0.18, 0.18) * tile_w)
            oy = int(random.uniform(-0.18, 0.18) * tile_h)
        else:
            ox = oy = 0
        place_x = max(0, min(canvas_w - tile_w, base_x + ox))
        place_y = max(0, min(canvas_h - tile_h, base_y + oy))

        # paste tile onto canvas (hard paste here; seams will be blended later)
        canvas[place_y:place_y + tile_h, place_x:place_x + tile_w] = tile_img
        offsets.append((place_x, place_y, new_w, new_h, xoff, yoff, iw, ih))

        # convert YOLO annotations to absolute coordinates in the canvas
        for a in anns:
            cls = int(a[0])
            x_c_rel, y_c_rel, bw_rel, bh_rel = float(a[1]), float(a[2]), float(a[3]), float(a[4])
            # original absolute in src_scaled
            # convert original rel coords (in original image) to pixel in scaled image
            ax = (x_c_rel * iw - xoff) * scale
            ay = (y_c_rel * ih - yoff) * scale
            a_w = bw_rel * iw * scale
            a_h = bh_rel * ih * scale
            # position in canvas
            abs_x = ax + place_x
            abs_y = ay + place_y
            # store
            quad_abs_ann.append((cls, abs_x, abs_y, a_w, a_h))

    # 3. seam blending
    canvas = blend_seams(canvas, tile_w, tile_h, seam_width=seam_width)

    # 4. object-centric or random crop
    chosen_keep = []
    x0 = y0 = 0
    for _ in range(max_retry):
        dx = int(random.uniform(-crop_offset, crop_offset) * tile_w)
        dy = int(random.uniform(-crop_offset, crop_offset) * tile_h)
        cx = tile_w + dx
        cy = tile_h + dy
        x0 = max(0, min(cx - tile_w // 2, canvas_w - tile_w))
        y0 = max(0, min(cy - tile_h // 2, canvas_h - tile_h))

        keep = []
        for (cls, ax, ay, aw, ah) in quad_abs_ann:
            if (x0 < ax < x0 + tile_w) and (y0 < ay < y0 + tile_h):
                keep.append((cls, ax, ay, aw, ah))
        # sometimes allow empty keep -> fallback to random crop
        if keep and random.random() < 0.9:
            chosen_keep = keep
            break
        elif not keep and random.random() < 0.3:
            chosen_keep = []
            break
    if not chosen_keep and len(quad_abs_ann) > 0:
        # fallback: pick any object and center on it
        obj = random.choice(quad_abs_ann)
        _, ax, ay, _, _ = obj
        x0 = int(max(0, min(ax - tile_w // 2, canvas_w - tile_w)))
        y0 = int(max(0, min(ay - tile_h // 2, canvas_h - tile_h)))
        chosen_keep = [o for o in quad_abs_ann if (x0 < o[1] < x0 + tile_w and y0 < o[2] < y0 + tile_h)]

    cropped = canvas[y0:y0 + tile_h, x0:x0 + tile_w].copy()

    # 5. update annotations to new cropped frame (YOLO relative)
    new_anns = []
    for (cls, ax, ay, aw, ah) in chosen_keep:
        x_center = (ax - x0) / float(tile_w)
        y_center = (ay - y0) / float(tile_h)
        bw = aw / float(tile_w)
        bh = ah / float(tile_h)
        if bw <= 0 or bh <= 0:
            continue
        if not (0 < x_center < 1 and 0 < y_center < 1):
            continue
        # clip box to image
        new_anns.append([cls, x_center, y_center, bw, bh])

    # 6. optional GridMask
    if random.random() < gridmask_prob and len(new_anns) > 0:
        if random.random() < 0.5:
            cropped = apply_gridmask(cropped, grid_ratio=grid_ratio)
        else:
            # simple cutout alternative: random rectangles
            for _ in range(random.randint(1, 3)):
                rw = random.randint(int(0.05 * tile_w), int(0.3 * tile_w))
                rh = random.randint(int(0.05 * tile_h), int(0.3 * tile_h))
                rx = random.randint(0, tile_w - rw)
                ry = random.randint(0, tile_h - rh)
                cropped[ry:ry + rh, rx:rx + rw] = 0
        # remove boxes heavily covered by mask (approx)
        kept = []
        for ann in new_anns:
            cls, x_c, y_c, bw, bh = ann
            cx = int(x_c * tile_w)
            cy = int(y_c * tile_h)
            box_w = max(1, int(bw * tile_w / 2))
            box_h = max(1, int(bh * tile_h / 2))
            x0b = max(0, cx - box_w)
            y0b = max(0, cy - box_h)
            x1b = min(tile_w, cx + box_w)
            y1b = min(tile_h, cy + box_h)
            patch = cropped[y0b:y1b, x0b:x1b]
            if patch.size == 0:
                continue
            # visible ratio heuristic: fraction of non-black pixels
            visible_ratio = np.count_nonzero(np.any(patch != 0, axis=-1)) / float(max(1, patch.shape[0] * patch.shape[1]))
            if visible_ratio > 0.25:
                kept.append(ann)
        new_anns = kept

    # 7. Copy-paste small objects (feathered + optional rotation/scale)
    if extra_images and extra_annotations and random.random() < copy_paste_prob and len(extra_annotations) > 0:
        ei = random.randrange(len(extra_images))
        src = extra_images[ei]
        eanns = extra_annotations[ei]
        if len(eanns) > 0 and len(new_anns) < 12:
            # paste 1..3 objects
            for _ in range(random.randint(1, 3)):
                a = random.choice(eanns)
                cls, x_c, y_c, bw, bh = int(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])
                ih, iw = src.shape[:2]
                x0o = int((x_c - bw / 2) * iw)
                y0o = int((y_c - bh / 2) * ih)
                w_obj = max(2, int(bw * iw))
                h_obj = max(2, int(bh * ih))
                if x0o < 0 or y0o < 0 or x0o + w_obj > iw or y0o + h_obj > ih:
                    continue
                patch = src[y0o:y0o + h_obj, x0o:x0o + w_obj].copy()
                # random scale/rotate patch
                scale_p = random.uniform(0.6, 1.2)
                pw = max(1, int(patch.shape[1] * scale_p))
                ph = max(1, int(patch.shape[0] * scale_p))
                patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
                if random.random() < 0.35:
                    angle = random.uniform(-25, 25)
                    M = cv2.getRotationMatrix2D((pw // 2, ph // 2), angle, 1.0)
                    patch = cv2.warpAffine(patch, M, (pw, ph), borderMode=cv2.BORDER_REFLECT)
                px = random.randint(0, tile_w - patch.shape[1])
                py = random.randint(0, tile_h - patch.shape[0])
                cropped = paste_with_feather(cropped, patch, px, py, feather=8, color_match=True)
                new_anns.append([cls, (px + patch.shape[1] / 2) / tile_w, (py + patch.shape[0] / 2) / tile_h,
                                 patch.shape[1] / tile_w, patch.shape[0] / tile_h])

    # 8. final global photometric harmonization
    cropped = global_photometric(cropped, prob=1.0)

    # 9. final sanitation: clip small boxes, ensure numeric types
    final_anns = []
    for ann in new_anns:
        cls, x_c, y_c, bw, bh = ann
        # discard boxes too small or out of range
        if bw <= 0 or bh <= 0:
            continue
        if bw * tile_w < 6 or bh * tile_h < 6:
            continue
        if not (0 < x_c < 1 and 0 < y_c < 1):
            continue
        final_anns.append([int(cls), float(x_c), float(y_c), float(bw), float(bh)])

    if len(final_anns) == 0:
        final_anns = np.zeros((0, 5), dtype=np.float32)
    else:
        final_anns = np.array(final_anns, dtype=np.float32)

    return cropped, final_anns


# --------------------------- Simple test helper ---------------------------

def draw_yolo_boxes(img: np.ndarray, anns: np.ndarray, class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for a in anns:
        cls, xc, yc, bw, bh = int(a[0]), a[1], a[2], a[3], a[4]
        x = int((xc - bw / 2) * w)
        y = int((yc - bh / 2) * h)
        ww = int(bw * w)
        hh = int(bh * h)
        color = (0, 255, 0) if class_colors is None else class_colors.get(cls, (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + ww, y + hh), color, 2)
        cv2.putText(out, str(cls), (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


if __name__ == '__main__':
    # quick sanity check (user should provide dataset to actually run)
    print('enhanced_mosaic_v2.py loaded. Integrate enhanced_mosaic_v2(images, annotations, class_freqs, ...)')
