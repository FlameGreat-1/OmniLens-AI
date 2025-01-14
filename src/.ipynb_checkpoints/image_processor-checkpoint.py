import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, restoration, segmentation, color
import logging

class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def enhance_image(self, image, contrast=1.2, brightness=10, saturation=1.2):
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge channels
            limg = cv2.merge((cl,a,b))
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Adjust contrast and brightness
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness)
            
            # Adjust saturation
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = hsv[:,:,1] * saturation
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return enhanced
        except Exception as e:
            self.logger.error(f"Error in enhance_image: {str(e)}")
            return image

    def denoise_image(self, image, method='nlm', strength=10):
        try:
            if method == 'nlm':
                return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
            elif method == 'bilateral':
                return cv2.bilateralFilter(image, 9, 75, 75)
            elif method == 'tv':
                return restoration.denoise_tv_chambolle(image, weight=0.1, multichannel=True)
            else:
                raise ValueError(f"Unknown denoising method: {method}")
        except Exception as e:
            self.logger.error(f"Error in denoise_image: {str(e)}")
            return image

    def sharpen_image(self, image, method='unsharp_mask', strength=1.0):
        try:
            if method == 'unsharp_mask':
                blurred = cv2.GaussianBlur(image, (0, 0), 3)
                return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            elif method == 'laplacian':
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                return cv2.filter2D(image, -1, kernel)
            else:
                raise ValueError(f"Unknown sharpening method: {method}")
        except Exception as e:
            self.logger.error(f"Error in sharpen_image: {str(e)}")
            return image

    def adjust_gamma(self, image, gamma=1.0):
        try:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        except Exception as e:
            self.logger.error(f"Error in adjust_gamma: {str(e)}")
            return image

    def correct_color_balance(self, image):
        try:
            result = exposure.equalize_adapthist(image, clip_limit=0.03)
            return (result * 255).astype(np.uint8)
        except Exception as e:
            self.logger.error(f"Error in correct_color_balance: {str(e)}")
            return image

    def remove_background(self, image):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            markers = cv2.watershed(image, markers)
            image[markers == -1] = [255, 0, 0]
            
            return image
        except Exception as e:
            self.logger.error(f"Error in remove_background: {str(e)}")
            return image

    def apply_artistic_filter(self, image, filter_type):
        try:
            if filter_type == "pencil_sketch":
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                inverted_image = 255 - gray_image
                blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
                inverted_blurred = 255 - blurred
                pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
                return cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2BGR)
            elif filter_type == "cartoon":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(image, 9, 300, 300)
                return cv2.bitwise_and(color, color, mask=edges)
            else:
                raise ValueError(f"Unknown artistic filter: {filter_type}")
        except Exception as e:
            self.logger.error(f"Error in apply_artistic_filter: {str(e)}")
            return image

    def resize_image(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        try:
            dim = None
            (h, w) = image.shape[:2]

            if width is None and height is None:
                return image

            if width is None:
                r = height / float(h)
                dim = (int(w * r), height)
            else:
                r = width / float(w)
                dim = (width, int(h * r))

            resized = cv2.resize(image, dim, interpolation=inter)
            return resized
        except Exception as e:
            self.logger.error(f"Error in resize_image: {str(e)}")
            return image

    def rotate_image(self, image, angle, center=None, scale=1.0):
        try:
            (h, w) = image.shape[:2]

            if center is None:
                center = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h))

            return rotated
        except Exception as e:
            self.logger.error(f"Error in rotate_image: {str(e)}")
            return image

    def crop_image(self, image, x, y, w, h):
        try:
            return image[y:y+h, x:x+w]
        except Exception as e:
            self.logger.error(f"Error in crop_image: {str(e)}")
            return image
