import os
import cv2
import matplotlib.pyplot as plt

# Set the current working directory as the base folder
base_dir = os.getcwd()  # Gets the directory where the script is run from

# Set paths relative to base_dir
image_dir = os.path.join(base_dir, 'valid/images')        
label_dir = os.path.join(base_dir, 'valid/labels')        
output_dir = os.path.join(base_dir, 'output_test')       

os.makedirs(output_dir, exist_ok=True)

# Class names list remains the same
class_names = ['BrownishRed--Can-be-Harvested', 'Green--Not-ready-to-harvest', 'GreenishYellow--Can-be-Harvested', 'Red-Ready-to-Harvest', 'Yellow-Ready-to-Harvest']

def draw_boxes(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    if not os.path.exists(label_path):
        print(f"No label for: {image_path}")
        return img

    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)

            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img

for image_file in os.listdir(image_dir):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

        result_img = draw_boxes(image_path, label_path)

        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, result_img)

        # Uncomment if you want to show images
        # plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        # plt.title(image_file)
        # plt.axis('off')
        # plt.show()
