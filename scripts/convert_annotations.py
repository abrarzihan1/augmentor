import os

def polygon_to_bbox(polygon_points):
    x_coords = polygon_points[::2]
    y_coords = polygon_points[1::2]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

def convert_annotation_file(file_path):
    new_lines = []

    with open(file_path, 'r') as infile:
        for line in infile:
            values = list(map(float, line.strip().split()))
            if len(values) < 3 or len(values) % 2 == 0:
                print(f"Skipping invalid line in {file_path}: {line}")
                continue
            class_id = int(values[0])
            polygon = values[1:]
            x_center, y_center, width, height = polygon_to_bbox(polygon)
            new_line = f"{class_id} {x_center:.10f} {y_center:.10f} {width:.10f} {height:.10f}\n"
            new_lines.append(new_line)

    with open(file_path, 'w') as outfile:
        outfile.writelines(new_lines)

def process_dataset(root_folder):
    subsets = ['train', 'test', 'valid']
    for subset in subsets:
        label_folder = os.path.join(root_folder, subset, 'labels_o')
        if not os.path.exists(label_folder):
            print(f"Labels folder not found: {label_folder}")
            continue

        for filename in os.listdir(label_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(label_folder, filename)
                convert_annotation_file(file_path)
                print(f"Converted: {file_path}")

# 🔧 Usage:
root_dataset_folder = 'pepper_detector'  # Replace with the path to your dataset root
process_dataset(root_dataset_folder)
