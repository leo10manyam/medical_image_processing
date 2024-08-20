import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the medical image
image_path = 'C:/Users/subra/OneDrive/Desktop/medicalimg/medicalradio2.jpeg'
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Image not found at the path: {image_path}")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram Equalization
equalized_image = cv2.equalizeHist(gray_image)

# Contour Detection
ret, thresh = cv2.threshold(equalized_image, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

# Color Mapping
color_mapped_image = cv2.applyColorMap(equalized_image, cv2.COLORMAP_JET)

# K-means Clustering
Z = image.reshape((-1, 3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
kmeans_image = res.reshape((image.shape))

# Morphological Operations
kernel = np.ones((3, 3), np.uint8)

# Applying morphological operations
dilated_image = cv2.dilate(thresh, kernel, iterations=1)
eroded_image = cv2.erode(thresh, kernel, iterations=1)
opened_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Display the results
titles = ['Original Image', 'Grayscale Image', 'Histogram Equalized', 'Contours',
          'Color Mapping', 'K-means Clustering', 'Dilation', 'Erosion', 'Opening', 'Closing']
images = [image, gray_image, equalized_image, contour_image,
          color_mapped_image, kmeans_image, dilated_image, eroded_image, opened_image, closed_image]

plt.figure(figsize=(15, 10))
for i in range(len(titles)):
    plt.subplot(2, 5, i+1), plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()