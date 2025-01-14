import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

def load_custom_dataset(dataset_path):
    images = []
    annotations = []
    
    image_dir = os.path.join(dataset_path, 'images')
    annot_dir = os.path.join(dataset_path, 'annotations')
    
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        ann_path = os.path.join(annot_dir, img_file.replace('.jpg', '.xml'))
        
        # Load image
        img = Image.open(img_path)
        images.append(np.array(img))
        
        # Parse annotation
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        classes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(name)
        
        annotations.append({'boxes': np.array(boxes), 'classes': np.array(classes)})
    
    return np.array(images), annotations

# Usage
images, annotations = load_custom_dataset('path/to/custom_dataset')
