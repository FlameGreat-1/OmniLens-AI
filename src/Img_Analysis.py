import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

class ImageAnalyzer:
    def get_histogram(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist

    def get_color_distribution(self, image):
        (b, g, r) = cv2.split(image)
        return {
            'blue': np.mean(b),
            'green': np.mean(g),
            'red': np.mean(r)
        }

    def get_metadata(self, image_path):
        image = Image.open(image_path)
        exif_data = {}
        info = image._getexif()
        if info:
            for tag_id, value in info.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
        return exif_data
