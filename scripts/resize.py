import os
from augmentor import utils
import cv2

image_dir = '../datasets/tomato/valid/images_o/'
output_img_dir = '../datasets/tomato/valid/images/'

os.makedirs(output_img_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in image_files:
    print(f'Resizing {image_file}')
    image = cv2.imread(f'{image_dir}/{image_file}')

    resized_image = utils.resize_image(image)

    cv2.imwrite(f'{output_img_dir}/{image_file}', resized_image)