import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def image_read(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    plt.show()
    return image


def equalize_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


def clahe_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def count_buildings(original_image, cont):
    total = 0
    for c in cont:
        area = cv2.contourArea(c)
        if 50 < area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)

            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)

            if 0.5 < aspect_ratio < 2.0 and 4 <= len(approx) <= 10:
                cv2.drawContours(original_image, [approx], -1, (0, 255, 0), 2)
                # cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                total += 1
    print(f"Знайдено {total} будівель")
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('Detected Buildings')
    plt.show()
    return total


# def segment_kmeans(image, K=4):
#     twoDimage = image.reshape((-1, 3))
#     twoDimage = np.float32(twoDimage)
#
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     attempts = 10
#     _, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
#
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     segmented_image = res.reshape((image.shape))
#     return segmented_image, label.reshape((image.shape[0], image.shape[1])), center
#
# def extract_building_cluster(segmented_labels, centers, cluster_index=None):
#     """Выбираем кластер будівель по цвету или по индексу"""
#     if cluster_index is None:
#         brightness = np.sum(centers, axis=1)
#         cluster_index = np.argmax(brightness)
#     mask = (segmented_labels == cluster_index).astype(np.uint8) * 255
#
#     # морфология для удаления шума
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     return mask


image = image_read('kpi_google_earth.png')

# segmented_image, labels, centers = segment_kmeans(image, K=2)
#
#
# plt.imshow(segmented_image, cmap='gray')
# plt.axis('off')
# plt.title('Segmented Image (K-means)')
# plt.show()
#
# mask = extract_building_cluster(labels, centers)
# plt.imshow(mask, cmap='gray')
# plt.axis('off')
# plt.title('Building Mask')
# plt.show()

image_better = equalize_histogram(image)
# image_better = clahe_enhancement(image)

# image_better = cv2.GaussianBlur(image_better, (5, 5), 0)

plt.imshow(image_better, cmap='gray')
plt.axis('off')
plt.title('Enhanced Image')
plt.show()

# _, binary = cv2.threshold(image_better, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(image_better, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               111, 2)

plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.title("After Adaptive Thresholding")
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
edges = cv2.bitwise_not(edges)

plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.title('Closed Edges')
plt.show()

contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_contours = image.copy()

cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
plt.figure(figsize=(10, 10))
plt.imshow(image_contours)
plt.axis('off')
plt.title('Contours')
plt.show()

count_buildings(image_contours, contours)
