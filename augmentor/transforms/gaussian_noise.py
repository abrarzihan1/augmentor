import numpy as np

def gaussian_noise(image, annotations=None, mean=0.0, std=20.0, **kwargs):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    if annotations is not None and annotations.size > 0:
        if annotations.ndim == 1:
            annotations = annotations.reshape(1, -1)
        return noisy_image, annotations
    return noisy_image, np.empty((0, 5))
