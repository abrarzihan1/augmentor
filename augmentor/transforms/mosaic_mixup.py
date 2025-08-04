import cv2
import numpy as np
import random

def mosaic_mixup(
    get_item_fn,
    dataset_size,
    output_size=(640, 640),
    crop_offset=0.3,
    mixup_prob=0.5,
    mixup_alpha=1.0,
):
    """
    Returns a function that generates a mosaic+mixup augmented image and labels.

    Args:
        get_item_fn (callable): function(idx) -> (image, annotations)
        dataset_size (int): total number of items in dataset.
        output_size (tuple): (width, height) of each final image.
        crop_offset (float): max center offset as a fraction of width/height.
        mixup_prob (float): probability to apply mixup after mosaic.
        mixup_alpha (float): alpha for Beta distribution used in MixUp.

    Returns:
        function(idx) -> (img, annotations)
    """
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
            for cls, x_ctr, y_ctr, bw, bh in anns_per_img:
                ax = x_ctr * w + ox
                ay = y_ctr * h + oy
                bbw, bbh = bw * w, bh * h
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
                out_anns.append([int(cls), ncx, ncy, nbw, nbh])

        return crop, np.array(out_anns, dtype=np.float32)

    def _transform(idx):
        img1, ann1 = _build_mosaic(idx)
        if random.random() < mixup_prob:
            img2, ann2 = _build_mosaic(random.randrange(dataset_size))
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            mix_img = (img1.astype(np.float32) * lam + img2.astype(np.float32) * (1 - lam)).astype(np.uint8)
            mix_anns = np.vstack([ann1, ann2])
            return mix_img, mix_anns
        else:
            return img1, ann1

    return _transform
