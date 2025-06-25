import albumentations as A
import numpy as np

def yolo_to_voc(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [x_min, y_min, x_max, y_max]

def voc_to_yolo(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]

def get_transform():
    return A.Compose(
        [A.Rotate(limit=90, p=1.0)],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.0  # don't filter anything
        )
    )

def apply(image, annotations=None, **kwargs):
    h, w = image.shape[:2]

    if annotations is not None and annotations.size > 0:
        annotations = np.array(annotations)
        if annotations.ndim == 1:
            annotations = np.expand_dims(annotations, axis=0)

        labels = annotations[:, 0]
        bboxes_yolo = annotations[:, 1:]

        bboxes_voc = []
        for b in bboxes_yolo:
            x_min, y_min, x_max, y_max = yolo_to_voc(b, w, h)

            # ✅ Clip to image bounds to prevent Albumentations errors
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # ✅ Only include boxes that still have positive area
            if x_max > x_min and y_max > y_min:
                bboxes_voc.append([x_min, y_min, x_max, y_max])
            else:
                print(f"Skipping invalid box: {x_min, y_min, x_max, y_max}")
    else:
        bboxes_voc = []
        labels = []

    aug = get_transform()

    try:
        transformed = aug(image=image, bboxes=bboxes_voc, class_labels=labels)
    except Exception as e:
        print("Error with bboxes_voc:")
        for b in bboxes_voc:
            print(b)
        raise e

    if len(transformed['bboxes']) > 0:
        bboxes_yolo = [voc_to_yolo(b, w, h) for b in transformed['bboxes']]
        bboxes_yolo = [np.clip(b, 0.0, 1.0) for b in bboxes_yolo]
        combined = np.array([
            [label, *bbox] for label, bbox in zip(transformed['class_labels'], bboxes_yolo)
        ])
    else:
        combined = np.array([])

    return transformed['image'], combined
