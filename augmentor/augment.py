from .transforms import (
    gamma,
    rotation,
    mosaic,
    translation,
    gaussian_noise,
    brightness_contrast,
    modified_mosaic,
    mixup,
    cutout
)

import numpy as np

# Internal mapping used by apply_transform
TRANSFORMS = {
    'gamma': gamma.apply,
    'rotation': rotation.apply,
    'translation': translation.apply,
    'noise': gaussian_noise.gaussian_noise,
    'brightness_contrast': brightness_contrast.apply,
    'mosaic': mosaic.mosaic,
    'modified_mosaic': modified_mosaic.mosaic,
    'mixup': mixup.apply,
    'cutout': cutout.apply,
}

multi_image_methods = ['mosaic', 'modified_mosaic', 'mixup']

def apply_transform(method, images, annotations=None, **kwargs):
    if method not in TRANSFORMS:
        raise ValueError(f"Unknown transform method: {method}")

    # Normalize input: single image -> [image]
    if method not in multi_image_methods and isinstance(images, np.ndarray):
        images = [images]
        annotations = [annotations]

    return TRANSFORMS[method](
        images if method in multi_image_methods else images[0],
        annotations if method in multi_image_methods else annotations[0],
        **kwargs
    )

gamma = gamma.apply
rotation = rotation.apply
translation = translation.apply
noise = gaussian_noise.gaussian_noise
brightness_contrast = brightness_contrast.apply
mosaic = mosaic.mosaic
modified_mosaic = modified_mosaic.mosaic
mixup = mixup.apply
cutout = cutout.apply
