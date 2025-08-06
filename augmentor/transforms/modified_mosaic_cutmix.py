import cv2
import numpy as np
import random
from . import brightness_contrast

def mosaic_cutmix(
    get_item_fn,
    dataset_size,
    output_size=(640, 640),
    crop_offset=0.15,
    cutmix_prob=0.1,
    brightness_contrast_prob=0.1,
    min_box_size=0.01,  # Normalized threshold (e.g., 0.01 = 6.4 pixels on 640)
):
    w, h = output_size

    def _build_mosaic(idx):
        ids = [idx] + random.choices(range(dataset_size), k=3)
        imgs, anns = zip(*(get_item_fn(i) for i in ids))
        tiles = [cv2.resize(im, output_size) for im in imgs]

        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        offsets = [(0, 0), (w, 0), (0, h), (w, h)]
        for tile, (ox, oy) in zip(tiles, offsets):
            canvas[oy:oy + h, ox:ox + w] = tile

        dx = int(random.uniform(-crop_offset, crop_offset) * w)
        dy = int(random.uniform(-crop_offset, crop_offset) * h)
        cx, cy = w + dx, h + dy
        x0, y0 = cx - w // 2, cy - h // 2
        crop = canvas[y0:y0 + h, x0:x0 + w]

        out_anns = []
        for anns_per_img, (ox, oy) in zip(anns, offsets):
            for cls, x, y, bw, bh in anns_per_img:
                ax = x * w + ox
                ay = y * h + oy
                bbw = bw * w
                bbh = bh * h
                x1 = ax - bbw / 2 - x0
                y1 = ay - bbh / 2 - y0
                x2 = ax + bbw / 2 - x0
                y2 = ay + bbh / 2 - y0

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                ncx = (x1 + x2) / (2 * w)
                ncy = (y1 + y2) / (2 * h)
                nbw = (x2 - x1) / w
                nbh = (y2 - y1) / h

                if nbw < min_box_size or nbh < min_box_size:
                    continue

                out_anns.append([int(cls), ncx, ncy, nbw, nbh])

        return crop, np.array(out_anns, dtype=np.float32)

    def _apply_cutmix(img1, anns1):
        idx2 = random.randint(0, dataset_size - 1)
        img2, anns2 = get_item_fn(idx2)
        img2 = cv2.resize(img2, output_size)

        # Select random patch region
        lam = np.random.beta(1.0, 1.0)
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply CutMix patch
        img1[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

        filtered_anns1 = []
        for cls, x, y, bw, bh in anns1:
            box_x1 = (x - bw / 2) * w
            box_y1 = (y - bh / 2) * h
            box_x2 = (x + bw / 2) * w
            box_y2 = (y + bh / 2) * h

            inter_x1 = max(x1, box_x1)
            inter_y1 = max(y1, box_y1)
            inter_x2 = min(x2, box_x2)
            inter_y2 = min(y2, box_y2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
            occlusion_ratio = inter_area / box_area if box_area > 0 else 0

            if occlusion_ratio < 0.5:
                filtered_anns1.append([cls, x, y, bw, bh])

        cutmix_anns = []
        for cls, x, y, bw, bh in anns2:
            box_x1 = (x - bw / 2) * w
            box_y1 = (y - bh / 2) * h
            box_x2 = (x + bw / 2) * w
            box_y2 = (y + bh / 2) * h

            inter_x1 = max(x1, box_x1)
            inter_y1 = max(y1, box_y1)
            inter_x2 = min(x2, box_x2)
            inter_y2 = min(y2, box_y2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                clipped_x1 = inter_x1
                clipped_y1 = inter_y1
                clipped_x2 = inter_x2
                clipped_y2 = inter_y2

                bw_new = (clipped_x2 - clipped_x1) / w
                bh_new = (clipped_y2 - clipped_y1) / h

                if bw_new < min_box_size or bh_new < min_box_size:
                    continue

                cx_new = (clipped_x1 + clipped_x2) / 2 / w
                cy_new = (clipped_y1 + clipped_y2) / 2 / h

                cutmix_anns.append([int(cls), cx_new, cy_new, bw_new, bh_new])

        anns_combined = np.vstack([
            np.array(filtered_anns1, dtype=np.float32).reshape(-1, 5),
            np.array(cutmix_anns, dtype=np.float32).reshape(-1, 5)
        ])

        return img1, anns_combined

    def _transform(idx):
        img, anns = _build_mosaic(idx)
        if random.random() < cutmix_prob:
            img, anns = _apply_cutmix(img, anns)
        if random.random() < brightness_contrast_prob:
            img, anns = brightness_contrast.apply(img, anns)
        return img, anns

    return _transform
