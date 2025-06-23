import cv2
import numpy as np
import random

def apply(image, annotations=None, **kwargs):
    brightness = random.uniform(-0.2, 0.2)
    contrast = random.uniform(-0.2, 0.2)

    beta = brightness * 255
    alpha = 1.0 + contrast

    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if annotations is not None and annotations.size > 0:
        if annotations.ndim == 1:
            annotations = annotations.reshape(1, -1)
        return new_image, annotations
    else:
        return new_image, np.empty((0, 5))
