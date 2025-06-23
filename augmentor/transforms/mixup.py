import numpy as np
import cv2

def yolo_to_voc(box, img_width, img_height):
    cls, xc, yc, w, h = box
    x1 = (xc - w / 2) * img_width
    y1 = (yc - h / 2) * img_height
    x2 = (xc + w / 2) * img_width
    y2 = (yc + h / 2) * img_height
    return [x1, y1, x2, y2, int(cls)]

def voc_to_yolo(box, img_width, img_height):
    x1, y1, x2, y2, cls = box
    xc = ((x1 + x2) / 2) / img_width
    yc = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return [int(cls), xc, yc, w, h]

def resize_image_and_boxes(image, boxes, target_size):
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    resized_img = cv2.resize(image, (target_w, target_h))

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    resized_boxes = []
    for box in boxes:
        x1, y1, x2, y2, cls = box
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        resized_boxes.append([x1, y1, x2, y2, cls])

    return resized_img, resized_boxes

def apply(images, annotations, image_size=(640, 640), alpha=1.0, **kwargs):
    """
    MixUp compatible with apply_transform:
    images_o: list of two np.ndarray images_o (RGB)
    annotations: list of two lists of YOLO boxes [class_id, x_center, y_center, w, h]

    Returns:
        mixed_image: np.ndarray
        mixed_annotations: np.ndarray of YOLO boxes
    """
    if len(images) < 2 or len(annotations) < 2:
        raise ValueError("MixUp requires at least two images_o and annotations")

    H, W = image_size

    # Convert YOLO -> VOC
    voc_boxes_0 = [yolo_to_voc(b, W, H) for b in annotations[0]]
    voc_boxes_1 = [yolo_to_voc(b, W, H) for b in annotations[1]]

    # Resize images_o and boxes
    img0, voc_boxes_0 = resize_image_and_boxes(images[0], voc_boxes_0, (W, H))
    img1, voc_boxes_1 = resize_image_and_boxes(images[1], voc_boxes_1, (W, H))

    # Sample lambda
    lam = np.random.beta(alpha, alpha)

    # Mix images_o
    mixed_img = (img0.astype(np.float32) * lam + img1.astype(np.float32) * (1 - lam)).astype(np.uint8)

    # Combine boxes
    mixed_boxes = voc_boxes_0 + voc_boxes_1

    # Clip boxes to image boundaries
    for box in mixed_boxes:
        box[0] = np.clip(box[0], 0, W)
        box[1] = np.clip(box[1], 0, H)
        box[2] = np.clip(box[2], 0, W)
        box[3] = np.clip(box[3], 0, H)

    # Convert VOC -> YOLO normalized
    mixed_annotations = [voc_to_yolo(b, W, H) for b in mixed_boxes]

    # Convert to NumPy array to allow slicing like [:, 0]
    mixed_annotations = np.array(mixed_annotations, dtype=np.float32)

    return mixed_img, mixed_annotations
