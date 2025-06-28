from augmentor import utils, augment
import cv2

img = cv2.imread("output_1/images/aug_29.jpg")
lbl = utils.load_yolo_annotation("output_1/labels/aug_29.txt")

aug_img, aug_lbl = img, lbl

img = utils.draw_image(aug_img, aug_lbl, class_names=['b_fully_ripened', 'b_half_ripened', 'b_green', 'l_fully_ripened', 'l_half_ripened', 'l_green'])
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()