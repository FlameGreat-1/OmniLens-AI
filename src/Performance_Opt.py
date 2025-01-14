import numpy as np
from numba import jit
import cv2

class PerformanceOptimizer:
    @staticmethod
    @jit(nopython=True)
    def fast_enhance(image, contrast, brightness):
        return np.clip(contrast * image + brightness, 0, 255).astype(np.uint8)

    @staticmethod
    def optimize_image_processing(image, operations):
        # Combine multiple operations into a single pass
        result = image.copy()
        for op in operations:
            if op['type'] == 'enhance':
                result = PerformanceOptimizer.fast_enhance(result, op['contrast'], op['brightness'])
            elif op['type'] == 'blur':
                result = cv2.GaussianBlur(result, op['kernel_size'], op['sigma'])
            # Add more operations as needed
        return result

    @staticmethod
    def parallel_process_images(images, operation, num_threads=4):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            return list(executor.map(operation, images))


