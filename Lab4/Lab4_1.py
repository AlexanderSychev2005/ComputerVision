import cv2

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')


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


def evaluate_matches(keypoints1, keypoints2, good_matches, matches_mask, matches, image1, image2):
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

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

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

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matches_mask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img_matches = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, **draw_params)

    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT comparing, probability: {probability:.3f}")
    plt.show()
    return probability



image1 = image_read('building_google_earth.png')
image2 = image_read('building_map-carta.png')

plt.figure(figsize=(10, 5))
plt.suptitle('Original Images')
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.axis('off')
plt.title('Image 1')
plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.axis('off')
plt.title('Image 2')
plt.show()

# image_better1 = gray_scale(image1)
# image_better2 = gray_scale(image2)
image_better1 = clahe_enhancement(image1)
image_better2 = clahe_enhancement(image2)

plt.figure(figsize=(10, 5))
plt.suptitle('Enhanced Images')
plt.subplot(1, 2, 1)
plt.imshow(image_better1, cmap='gray')
plt.axis('off')
plt.title('Image 1')
plt.subplot(1, 2, 2)
plt.imshow(image_better2, cmap='gray')
plt.axis('off')
plt.title('Image 2')
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



keypoints1, keypoints2, good_matches, matches_mask, matches = find_and_match_keypoints(image_better1, image_better2)
probability = evaluate_matches(keypoints1, keypoints2, good_matches, matches_mask, matches, image1, image2)
