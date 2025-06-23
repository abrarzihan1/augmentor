import numpy as np
import cv2


def apply(image, annotations=None, n_holes=1, max_size=0.3, fill_value=0, min_iou=0.5, **kwargs):
    """
    Applies Cutout augmentation on a single image.

    Args:
        image (np.ndarray): Input image (HWC, RGB)
        annotations (list of [class, xc, yc, w, h]): YOLO normalized format
        n_holes (int): Number of cutout holes
        max_size (float): Maximum size of cutout region (as a fraction of image)
        fill_value (int or tuple): Value to fill cutout with (e.g. 0 for black)
        min_iou (float): Minimum IoU threshold to keep a box (optional)

    Returns:
        image: Augmented image
        annotations: Filtered YOLO boxes (np.ndarray)
    """
    h, w = image.shape[:2]

    if annotations is None:
        annotations = []

    # Convert YOLO -> VOC (absolute)
    voc_boxes = []
    for box in annotations:
        cls, xc, yc, bw, bh = box
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        voc_boxes.append([x1, y1, x2, y2, int(cls)])

    # Generate cutout holes
    for _ in range(n_holes):
        hole_w = int(np.random.uniform(0.1, max_size) * w)
        hole_h = int(np.random.uniform(0.1, max_size) * h)
        x1 = np.random.randint(0, w - hole_w)
        y1 = np.random.randint(0, h - hole_h)
        x2 = x1 + hole_w
        y2 = y1 + hole_h

        # Apply cutout
        image[y1:y2, x1:x2] = fill_value

        # Optionally remove boxes that intersect too much with cutout
        new_voc_boxes = []
        for box in voc_boxes:
            bx1, by1, bx2, by2, cls = box

            # Intersection
            inter_x1 = max(x1, bx1)
            inter_y1 = max(y1, by1)
            inter_x2 = min(x2, bx2)
            inter_y2 = min(y2, by2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            box_area = max(0, bx2 - bx1) * max(0, by2 - by1)
            iou = inter_area / box_area if box_area > 0 else 0

            if iou < min_iou:
                new_voc_boxes.append(box)
        voc_boxes = new_voc_boxes

    # Convert VOC -> YOLO
    final_annotations = []
    for box in voc_boxes:
        x1, y1, x2, y2, cls = box
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        final_annotations.append([cls, xc, yc, bw, bh])

    return image, np.array(final_annotations, dtype=np.float32)
