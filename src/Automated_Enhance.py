import cv2
import numpy as np
from skimage import exposure, restoration
from scipy import ndimage

class AutoEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def auto_enhance(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        cl = self.clahe.apply(l)

        # Apply bilateral filter for edge-preserving smoothing
        cl_smooth = cv2.bilateralFilter(cl, d=5, sigmaColor=10, sigmaSpace=10)

        # Enhance local contrast
        cl_enhanced = self.enhance_local_contrast(cl_smooth)

        # Merge channels
        enhanced_lab = cv2.merge((cl_enhanced, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Apply sharpening
        enhanced_image = self.sharpen_image(enhanced_image)

        # Apply color balance
        enhanced_image = self.color_balance(enhanced_image)

        return enhanced_image

    def enhance_local_contrast(self, image):
        # Enhance local contrast using CLAHE
        return self.clahe.apply(image)

    def sharpen_image(self, image):
        # Create a sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # Apply the kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def color_balance(self, image):
        # Simple color balance using percentile
        for i in range(3):
            channel = image[:,:,i]
            low, high = np.percentile(channel, (2, 98))
            channel = exposure.rescale_intensity(channel, in_range=(low, high))
            image[:,:,i] = channel
        return image

    def auto_adjust_gamma(self, image):
        # Estimate optimal gamma value
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mid = 0.5
        mean = np.mean(gray)
        gamma = np.log(mid*255)/np.log(mean)
        
        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def denoise_image(self, image):
        # Apply non-local means denoising
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def adjust_brightness_contrast(self, image, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
        else:
            buf = image.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def enhance_image(self, image):
        # Apply a series of enhancements
        enhanced = self.auto_enhance(image)
        enhanced = self.denoise_image(enhanced)
        enhanced = self.auto_adjust_gamma(enhanced)
        enhanced = self.adjust_brightness_contrast(enhanced, brightness=10, contrast=10)
        return enhanced

# Usage example:
# enhancer = AutoEnhancer()
# image = cv2.imread('input_image.jpg')
# enhanced_image = enhancer.enhance_image(image)
# cv2.imwrite('enhanced_image.jpg', enhanced_image)
