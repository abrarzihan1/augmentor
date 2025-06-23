import cv2
import numpy as np
from augmentor import augment, utils

images = [cv2.imread(f'../dataset/images/img_{i}.jpg') for i in range(1,5)]
annotations = [np.loadtxt(f'../dataset/labels/img_{i}.txt') for i in range(1,5)]

# Apply mosaic transform
aug_image, aug_annotations = augment.mosaic(images, annotations)
img = utils.draw_image(aug_image, aug_annotations, ['plane'])
cv2.imshow("augmented image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
