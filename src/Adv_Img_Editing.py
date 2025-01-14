import cv2
import numpy as np

class ImageProcessor:
    def enhance_image(self, image, contrast=1.2, brightness=50):
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    def sharpen_image(self, image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    def denoise_image(self, image):
        return cv2.fastNlMeansDenoisingColored(image)

    def crop_image(self, image, x, y, w, h):
        return image[y:y+h, x:x+w]

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def resize_image(self, image, width, height):
        return cv2.resize(image, (width, height))
