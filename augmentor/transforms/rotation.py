import albumentations as A
import numpy as np

def get_transform():
    return A.Compose(
        [A.Rotate(limit=90, p=1.0)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )

def apply(image, annotations=None, **kwargs):
    if annotations is not None and annotations.size > 0:
        annotations = np.array(annotations)  # Ensure it's a NumPy array
        if annotations.ndim == 1:
            annotations = np.expand_dims(annotations, axis=0)

        bboxes = annotations[:, 1:]
        labels = annotations[:, 0]
    else:
        bboxes = []
        labels = []

    aug = get_transform()
    transformed = aug(image=image, bboxes=bboxes, class_labels=labels)

    # Clip bboxes to [0.0, 1.0]
    transformed_bboxes = np.clip(transformed['bboxes'], 0.0, 1.0)

    # Combine class labels_o and bboxes
    if len(transformed['bboxes']) > 0:
        combined = np.array([
            [label, *bbox] for label, bbox in zip(transformed['class_labels'], transformed['bboxes'])
        ])
    else:
        combined = np.array([])

    return transformed['image'], combined
