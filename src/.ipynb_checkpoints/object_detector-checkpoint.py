import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import logging

class ObjectDetector:
    def __init__(self, model_path, label_map_path, num_classes):
        self.logger = logging.getLogger(__name__)
        try:
            # Load model
            self.detect_fn = tf.saved_model.load(model_path)

            # Load label map
            self.category_index = label_map_util.create_category_index_from_labelmap(
                label_map_path, use_display_name=True)

            self.num_classes = num_classes
        except Exception as e:
            self.logger.error(f"Error initializing ObjectDetector: {str(e)}")
            raise

    def detect_objects(self, image, min_score_thresh=0.5):
        try:
            # Convert image to tensor
            input_tensor = tf.convert_to_tensor(image)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Perform detection
            detections = self.detect_fn(input_tensor)

            # Process detections
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            # Filter detections based on minimum score threshold
            indices = np.where(detections['detection_scores'] >= min_score_thresh)[0]
            filtered_boxes = detections['detection_boxes'][indices]
            filtered_classes = detections['detection_classes'][indices]
            filtered_scores = detections['detection_scores'][indices]

            return filtered_boxes, filtered_classes, filtered_scores
        except Exception as e:
            self.logger.error(f"Error in detect_objects: {str(e)}")
            return [], [], []

    def visualize_detections(self, image, boxes, classes, scores):
        try:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image,
                boxes,
                classes,
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=0.5,
                agnostic_mode=False)
            return image
        except Exception as e:
            self.logger.error(f"Error in visualize_detections: {str(e)}")
            return image

    def track_objects(self, video_path, output_path):
        try:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            tracker = cv2.MultiTracker_create()
            init_tracking = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if not init_tracking:
                    boxes, classes, scores = self.detect_objects(frame)
                    for box in boxes:
                        tracker.add(cv2.TrackerKCF_create(), frame, box)
                    init_tracking = True
                else:
                    success, boxes = tracker.update(frame)
                    if success:
                        for i, new_box in enumerate(boxes):
                            p1 = (int(new_box[0]), int(new_box[1]))
                            p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
                            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                            cv2.putText(frame, f"Object {i}", (p1[0], p1[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"Error in track_objects: {str(e)}")

    def detect_and_classify(self, image, min_score_thresh=0.5):
        try:
            boxes, classes, scores = self.detect_objects(image, min_score_thresh)
            results = []
            for i in range(len(boxes)):
                class_name = self.category_index[classes[i]]['name']
                results.append({
                    'class': class_name,
                    'score': scores[i],
                    'box': boxes[i]
                })
            return results
        except Exception as e:
            self.logger.error(f"Error in detect_and_classify: {str(e)}")
            return []
