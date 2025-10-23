import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import defaultdict
from glob import glob
from pathlib import Path
import shutil


def extract_color_features(image_path, bins=(8, 8, 8)):
    """
    Extract color histogram features from an image.
    """
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute the 3D color histogram and normalize it
        hist = cv2.calcHist(
            [image_rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)  # Normalize the histogram, the sum of bins equals 1

        return hist.flatten()  # Flatten the histogram into a feature vector, 8 * 8 * 8 = 512 dimensions
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


def clustering_images(features, K):
    """
    Cluster images based on extracted features using KMeans.
    """
    x = np.array(features)

    kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(x)
    labels = kmeans.labels_
    return labels


dataset_path = "image_dataset"

K = 3
features_list = []
image_paths_list = []

valid_image_extensions = (".jpg", ".jpeg", ".png")
for folder in glob(os.path.join(dataset_path, "*")):
    for image_file in glob(os.path.join(folder, "*")):
        print(f"Processing image: {image_file}")
        if image_file.lower().endswith(valid_image_extensions):
            features = extract_color_features(image_file)
            if features is not None:
                features_list.append(features)
                image_paths_list.append(image_file)

if not features_list:
    print("No valid images found in the dataset path.")
    exit()

print(f"Extracted features from {len(features_list)} images.")
print(f"Dimension of features for each image: {features_list[0].shape}")

labels = clustering_images(features_list, K)

results_path = "clustering_results"
if os.path.exists(results_path):
    shutil.rmtree(results_path)

Path(results_path).mkdir(parents=True, exist_ok=True)
clustered_images = defaultdict(list)

for img_path, label in zip(image_paths_list, labels):
    clustered_images[label].append(img_path)

for cluster_id, img_paths in clustered_images.items():
    cluster_folder = os.path.join(results_path, f"cluster_{cluster_id}")
    Path(cluster_folder).mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        dest_path = os.path.join(cluster_folder, img_name)
        cv2.imwrite(dest_path, cv2.imread(img_path))
