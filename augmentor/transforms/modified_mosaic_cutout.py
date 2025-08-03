import random
from . import cutout
from . import mosaic

def apply(images, annotations):
    image, annotations = mosaic.mosaic(images, annotations)

    random_num = random.random()
    if random_num < 0.4:
        image, annotations = cutout.apply(image, annotations, n_holes=1)
    elif random_num < 0.7:
        image, annotations = cutout.apply(image, annotations, n_holes=2)

    return image, annotations
