import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from shapely.geometry import Polygon

matplotlib.use('TkAgg')


def image_read(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    plt.show()
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
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


def median_blur(image):
    return cv2.medianBlur(image, 3)


def segment_buildings(gray, method='adaptive'):
    if method == 'adaptive':
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 221, 6)
    else:
        raise ValueError("Method must be 'adaptive' or 'otsu' or 'canny'")

    plt.imshow(bin_img, cmap='gray')
    plt.axis('off')
    plt.title('Binary Image')
    plt.show()

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_open)
    bin_img = cv2.bitwise_not(bin_img)
    return bin_img


def find_contours(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = []
    h, w = bin_img.shape
    min_area = 100
    max_area = h * w * 0.5
    approx_factor = 0.02

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        epsilon = approx_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if approx.shape[0] < 3:
            continue

        coords = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        poly = Polygon(coords)
        if not poly.is_valid:
            continue

        bbox = poly.bounds  # minx, miny, maxx, maxy
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox[2] > bbox[0] and bbox[3] > bbox[1] else 1
        solidity = poly.area / bbox_area
        if solidity < 0.2:
            continue
        conts.append(poly)
    return conts


image = image_read('kpi_google_earth.png')


# image_filtered = bilateral_filter(image)

# image_better = gray_scale(image_filtered)

# image_better = equalize_histogram(image)
image_better = clahe_enhancement(image)
# image_better = median_blur(image_better)
# image_filtered = gaussian_blur(image_better)


plt.imshow(image_better, cmap='gray')
plt.axis('off')
plt.title('Enhanced Image')
plt.show()


bin_img = segment_buildings(image_better, method='adaptive')
plt.imshow(bin_img, cmap='gray')
plt.axis('off')
plt.title('Segmented Buildings')
plt.show()

contours = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
plt.imshow(image_contours)
plt.axis('off')
plt.title('Detected Building Contours')
plt.show()

contours = find_contours(bin_img)
num_buildings = len(contours)
print(f"Detected buildings: {num_buildings}")

image_contours = image.copy()
for poly in contours:
    coords = [(int(x), int(y)) for x,y in poly.exterior.coords]
    cv2.polylines(image_contours, [np.array(coords, dtype=np.int32)], True, (255, 0, 0), 2)
plt.imshow(image_contours)
plt.axis('off')
plt.title('Detected Building Contours')
plt.show()
