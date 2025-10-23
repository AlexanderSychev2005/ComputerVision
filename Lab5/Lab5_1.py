import cv2

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from skimage import filters
from skimage.util import img_as_float, img_as_ubyte

matplotlib.use("TkAgg")


def image_read(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (800, 500))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def gray_scale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray


def equalize_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


def clahe_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def bilateral_filter(image):
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered


def gaussian_blur(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    return blurred


def median_blur(image):
    return cv2.medianBlur(image, 3)


def find_and_match_keypoints(image1, image2):
    """
    Find and match key points between two images using SIFT detector.
    """
    detector = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        return ValueError("No descriptors found in one of the images.")

    # Creating an object for matching
    # FLANN (Fast Library for Approximate Nearest Neighbors) based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Using KNN algorithm to find the best matches for each descriptor
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches, matchesMask, matches


def evaluate_matches(
        keypoints1, keypoints2, good_matches, matches_mask, matches, image1, image2
):
    """
    Evaluate the quality of matches using RANSAC and calculate the probability of a match being correct.
    """
    if len(good_matches) < 10:
        print(f"Not enough good matches found {len(good_matches)}. Need at least 10.")
        return

    # Probability of a match being correct
    # The ratio of detected special points (good matches) to all detected points, all keypoints

    # len (keypoints1) - total keypoints in image 1
    # len (keypoints2) - total keypoints in image 2
    # len (matches) - good matches between the two images

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    H, ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)

    ransac_mask_list = ransac_mask.ravel().tolist()
    inliers_count = np.sum(ransac_mask_list)

    # Probability of a match being correct, number of inliers to total good matches
    probability = inliers_count / len(good_matches)
    print(f"Number of good matches: {len(good_matches)}")
    print(f"Number of inliers: {inliers_count}")
    print(f"Probability of a match being correct: {probability:.2f}")

    if probability > 0.5:
        print("The images are similar.")
    else:
        print("The images are not similar.")

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    img_matches = cv2.drawMatchesKnn(
        image1, keypoints1, image2, keypoints2, matches, None, **draw_params
    )

    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT comparing, probability: {probability:.3f}")
    plt.show()
    return probability


def segment_kmeans(image, k):
    """
    Segment the image using K-means clustering. Flatten all pixels and find k main colors. Algorithm finds
    3 centres of clusters and assigns each pixel to the nearest cluster center. This simplifies the image, leaving
    only the main colors.
    """
    # all pixels in a row, with r,g,b values
    pixel_values = image.reshape((-1, 3))
    print(pixel_values.shape)
    pixel_values = np.float32(pixel_values)

    # stop criteria, 100 iterations or epsilon 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # applying kmeans, returns centers (k colors) and labels (which pixel belongs to which cluster)
    compactness, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)

    # recreate the segmented image, applying the average color of the cluster to each pixel
    segmented_data = centers[labels.flatten()]

    # reshape back to the original image dimensions
    segmented_image = segmented_data.reshape(image.shape)

    # reshape labels to the original image dimensions, useful for visualization
    labels_image = labels.reshape(image.shape[:2])

    return segmented_image, labels_image


def segment_otsu(image):
    """
    Segment the image using Otsu's thresholding method combined with the Watershed algorithm.
    First, Otsu's method is applied to split the image into white and black regions. Then, morphological operations
    are used to find sure background and foreground areas.
    Finally, the Watershed algorithm is applied to find a border between different segments, where they meet.
    """
    gray = gray_scale(image)

    # Apply Otsu's thresholding, which automatically finds the optimal threshold value
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to close gaps
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Finding sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=3)

    # Finding sure foreground area
    # the more far the pixel is from the background, the more certain we are that it is foreground
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Finding unknown region, where the foreground and background meet
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling foreground regions
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # so that sure background is not 0, but 1
    markers[unknown == 255] = 0  # mark the unknown region with zero

    markers = cv2.watershed(image, markers)  # apply watershed algorithm, seeking borders between segments
    image[markers == -1] = [0, 0, 255]  # mark the borders with blue color

    return image


def edge_roberts(image):
    """
    Detect edges in the image using one of the most simple and old filter, Roberts Cross operator. It looks for
    changes in intensity along diagonal directions.
    """
    gray = gray_scale(image)

    gray_float = img_as_float(gray)
    roberts_edges = filters.roberts(gray_float)
    roberts_edges_uint8 = img_as_ubyte(roberts_edges)
    return roberts_edges_uint8


image1 = image_read("building_google_earth.png")
image2 = image_read("building_map-carta.png")

plt.figure(figsize=(10, 5))
plt.suptitle("Original Images")
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.axis("off")
plt.title("Image 1")
plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.axis("off")
plt.title("Image 2")
plt.show()

# image_better1 = gray_scale(image1)
# image_better2 = gray_scale(image2)
image_better1 = clahe_enhancement(image1)
image_better2 = clahe_enhancement(image2)

plt.figure(figsize=(10, 5))
plt.suptitle("Enhanced Images")
plt.subplot(1, 2, 1)
plt.imshow(image_better1, cmap="gray")
plt.axis("off")
plt.title("Image 1")
plt.subplot(1, 2, 2)
plt.imshow(image_better2, cmap="gray")
plt.axis("off")
plt.title("Image 2")
plt.show()

# image_better1 = gaussian_blur(image_better1)
# image_better2 = gaussian_blur(image_better2)

# plt.figure(figsize=(10, 5))
# plt.suptitle('Filtered Images')
# plt.subplot(1, 2, 1)
# plt.imshow(image_better1)
# plt.axis('off')
# plt.title('Image 1')
# plt.subplot(1, 2, 2)
# plt.imshow(image_better2)
# plt.axis('off')
# plt.title('Image 2')
# plt.show()


keypoints1, keypoints2, good_matches, matches_mask, matches = find_and_match_keypoints(
    image_better1, image_better2
)
probability = evaluate_matches(
    keypoints1, keypoints2, good_matches, matches_mask, matches, image1, image2
)

K = 3
segmented_image1, labels_image1 = segment_kmeans(image1, k=K)
segmented_image2, labels_image2 = segment_kmeans(image2, k=K)

plt.figure(figsize=(15, 12))
plt.suptitle(f"K-means Segmentation Results with k={K}")

plt.subplot(3, 2, 1)
plt.imshow(image1)
plt.axis("off")
plt.title("Original Image 1")

plt.subplot(3, 2, 2)
plt.imshow(image2)
plt.axis("off")
plt.title("Original Image 2")

plt.subplot(3, 2, 3)
plt.imshow(segmented_image1)
plt.axis("off")
plt.title("Segmented Image 1 (K-means)")

plt.subplot(3, 2, 4)
plt.imshow(segmented_image2)
plt.axis("off")
plt.title("Segmented Image 2 (K-means)")

plt.subplot(3, 2, 5)
plt.imshow(labels_image1, cmap="viridis")
plt.axis("off")
plt.title("Labels Image 1")

plt.subplot(3, 2, 6)
plt.imshow(labels_image2, cmap="viridis")
plt.axis("off")
plt.title("Labels Image 2")

plt.tight_layout()
plt.show()

otsu_segmented1 = segment_otsu(image1.copy())
otsu_segmented2 = segment_otsu(image2.copy())
plt.figure(figsize=(10, 5))
plt.suptitle("Otsu's Segmentation Results")
plt.subplot(1, 2, 1)
plt.imshow(otsu_segmented1)
plt.axis("off")
plt.title("Otsu's Segmented Image 1")
plt.subplot(1, 2, 2)
plt.imshow(otsu_segmented2)
plt.axis("off")
plt.title("Otsu's Segmented Image 2")
plt.show()

roberts_edges1 = edge_roberts(image1)
roberts_edges2 = edge_roberts(image2)
plt.figure(figsize=(10, 5))
plt.suptitle("Roberts Edge Detection Results")
plt.subplot(1, 2, 1)
plt.imshow(roberts_edges1, cmap="gray")
plt.axis("off")
plt.title("Roberts Edges Image 1")
plt.subplot(1, 2, 2)
plt.imshow(roberts_edges2, cmap="gray")
plt.axis("off")
plt.title("Roberts Edges Image 2")
plt.show()
