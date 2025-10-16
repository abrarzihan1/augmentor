from augmentor import utils, augment
import cv2

img = cv2.imread(r"C:\Users\User\Documents\Studies\Thesis\project\augmentor\test\augmented\images\aug_50.jpg")
lbl = utils.load_yolo_annotation(r"C:\Users\User\Documents\Studies\Thesis\project\augmentor\test\augmented\labels\aug_50.txt")

aug_img, aug_lbl = img, lbl

img = utils.draw_image(aug_img, aug_lbl, class_names=['Carpetweed', 'CutleafGroundcherry', 'Eclipta', 'Goosegrass', 'MorningGlory', 'PalmerAmaranth', 'PricklySida', 'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge', 'Waterhemp'])
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()