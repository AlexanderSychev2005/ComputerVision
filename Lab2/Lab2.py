import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from skimage.metrics import structural_similarity


def image_read(file_path1, file_path2):
    image1 = cv2.imread(file_path1)
    image2 = cv2.imread(file_path2)
    image1 = cv2.resize(image1, (800, 500))
    image2 = cv2.resize(image2, (800, 500))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    plt.imshow(image1)
    plt.axis('off')
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    plt.imshow(image2)
    plt.axis('off')
    plt.title('Image 2')
    plt.show()
    return image1, image2


def compare_image(edges1, edges2):
    ssim_score = structural_similarity(edges1, edges2)
    return ssim_score


def draw_contours(image1, image2, edges1, edges2):
    contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img1 = image1.copy()
    contour_img2 = image2.copy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    cv2.drawContours(contour_img1, contours1, -1, (0, 255, 0), 2)

    plt.imshow(contour_img1)
    plt.axis('off')
    plt.title('Contours Image 1')

    plt.subplot(1, 2, 2)

    cv2.drawContours(contour_img2, contours2, -1, (0, 255, 0), 2)
    plt.imshow(contour_img2)
    plt.axis('off')
    plt.title('Contours Image 2')
    plt.show()
    return contour_img1, contour_img2


def analyse_images(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)



    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray1, cmap='gray')
    plt.axis('off')
    plt.title('Image 1 Grayscale')
    plt.subplot(1, 2, 2)
    plt.imshow(gray2, cmap='gray')
    plt.axis('off')
    plt.title('Image 2 Grayscale')
    plt.show()

    gray1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray1)
    gray2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray2)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray1, cmap='gray')
    plt.axis('off')
    plt.title('Image 1 CLAHE')
    plt.subplot(1, 2, 2)
    plt.imshow(gray2, cmap='gray')
    plt.axis('off')
    plt.title('Image 2 CLAHE')
    plt.show()



    blur1 = cv2.GaussianBlur(gray1, (7, 7), 0)
    blur2 = cv2.GaussianBlur(gray2, (7, 7), 0)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(blur1, cmap='gray')
    plt.axis('off')
    plt.title('Image 1 Gaussian Blur')
    plt.subplot(1, 2, 2)
    plt.imshow(blur2, cmap='gray')
    plt.axis('off')
    plt.title('Image 2 Gaussian Blur')
    plt.show()

    edges1 = cv2.Canny(blur1, 120, 200)
    edges2 = cv2.Canny(blur2, 200, 370)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(edges1, cmap='gray')
    plt.axis('off')
    plt.title('Image 1 Canny Edges')
    plt.subplot(1, 2, 2)
    plt.imshow(edges2, cmap='gray')
    plt.axis('off')
    plt.title('Image 2 Canny Edges')
    plt.show()

    return edges1, edges2


satellite_img1, satellite_img2 = image_read('planes1.png', 'planes_light_back.png')

edges1, edges2 = analyse_images(satellite_img1, satellite_img2)

contour_img1 = draw_contours(satellite_img1, satellite_img2, edges1, edges2)
ssim_score = compare_image(edges1, edges2)
print(f'SSIM score between the two images: {ssim_score}')
