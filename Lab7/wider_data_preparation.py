import os
import cv2

IMAGE_BASE_PATH = "./images/train"
ANNOTATIONS_FILE = "./wider_face_train_bbx_gt.txt"
OUTPUT_DIR = "./labels/train"


os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ANNOTATIONS_FILE) as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    file_path = lines[i].strip()
    if '.jpg' not in file_path:
        i += 1
        continue


    full_image_path = os.path.join(IMAGE_BASE_PATH, file_path)

    sub_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)

    label_file_name = base_name.replace(".jpg", ".txt")
    label_output_dir = os.path.join(OUTPUT_DIR, sub_dir)
    os.makedirs(label_output_dir, exist_ok=True)

    label_output_path = os.path.join(label_output_dir, label_file_name)

    try:
        img = cv2.imread(full_image_path)
        if img is None:
            raise ValueError("Image not found or unable to read.")
        img_h, img_w, _ = img.shape
    except Exception as e:
        print(f"Error reading image {full_image_path}, {e}")

        try:
            num_faces = int(lines[i + 1].strip())
        except ValueError:
            i += 1
            continue
        i += 2 + num_faces
        continue

    num_faces = int(lines[i + 1].strip())
    i += 2

    yolo_labels = []
    for j in range(num_faces):
        bbox_info = lines[i + j].strip().split()
        x, y, w, h = map(int, bbox_info[:4])

        x_center = x + w / 2
        y_center = y + h / 2
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h

        width_norm = w / img_w
        height_norm = h / img_h

        yolo_labels.append(f"0 {x_center_norm} {y_center_norm} {width_norm} {height_norm}")
    if yolo_labels:
        with open(label_output_path, "w") as label_file:
            label_file.write("\n".join(yolo_labels))
    i += num_faces

