import os

# Define the paths to your 'images_o' and 'labels_o' directories
images_dir = '../datasets/pepper_detector/train/images'
labels_dir = '../datasets/pepper_detector/train/labels'

# Get the list of image files and label files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

# Ensure the number of image and label files match
if len(image_files) != len(label_files):
    print("Error: The number of image files and label files do not match.")
else:
    # Iterate over image files and rename them sequentially
    for idx, (img_file, label_file) in enumerate(zip(image_files, label_files), start=1):
        # Generate the new names for the files
        new_img_name = f'img_{idx}.jpg'
        new_label_name = f'img_{idx}.txt'
        
        # Get the full file paths
        old_img_path = os.path.join(images_dir, img_file)
        old_label_path = os.path.join(labels_dir, label_file)
        
        new_img_path = os.path.join(images_dir, new_img_name)
        new_label_path = os.path.join(labels_dir, new_label_name)
        
        # Rename the files
        os.rename(old_img_path, new_img_path)
        os.rename(old_label_path, new_label_path)
        
        print(f'Renamed: {img_file} -> {new_img_name}')
        print(f'Renamed: {label_file} -> {new_label_name}')

print("Renaming complete.")
