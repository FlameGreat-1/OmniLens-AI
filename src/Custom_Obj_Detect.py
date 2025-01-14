import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt

class ObjectDetector:
    def __init__(self, model_path, label_map_path, config_path):
        self.model = tf.saved_model.load(model_path)
        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
        self.configs = config_util.get_configs_from_pipeline_file(config_path)
        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)

    def detect_objects(self, image):
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.model(input_tensor)
        return detections

    def visualize_detections(self, image, detections):
        image_np = image.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
        plt.figure(figsize=(12,8))
        plt.imshow(image_np)
        plt.show()

    def fine_tune_model(self, dataset, epochs=10, batch_size=32, learning_rate=0.001):
        # Prepare the dataset
        train_dataset = self.prepare_dataset(dataset, batch_size)

        # Get the model ready for fine-tuning
        self.detection_model.load_weights(self.configs['model'].model.checkpoint_path)
        variables_to_fine_tune = self.detection_model.trainable_variables
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                preprocessed_images = self.detection_model.preprocess(images)
                prediction_dict = self.detection_model.predict(preprocessed_images, labels)
                losses_dict = self.detection_model.loss(prediction_dict, labels)
                total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, variables_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, variables_to_fine_tune))
            return total_loss

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            total_loss = 0
            num_batches = 0
            for batch_images, batch_labels in train_dataset:
                loss = train_step(batch_images, batch_labels)
                total_loss += loss
                num_batches += 1
            average_loss = total_loss / num_batches
            print(f"Average loss: {average_loss:.4f}")

        # Save the fine-tuned model
        self.detection_model.save_weights('fine_tuned_model_weights')
        print("Fine-tuning complete. Model weights saved.")

    def prepare_dataset(self, dataset, batch_size):
        def parse_tfrecord(serialized_example):
            feature_description = {
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/object/class/label': tf.io.VarLenFeature(tf.int64)
            }
            example = tf.io.parse_single_example(serialized_example, feature_description)
            
            image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
            image = tf.image.resize(image, [300, 300])
            image = tf.cast(image, tf.float32) / 255.0
            
            xmins = tf.sparse.to_dense(example['image/object/bbox/xmin'])
            xmaxs = tf.sparse.to_dense(example['image/object/bbox/xmax'])
            ymins = tf.sparse.to_dense(example['image/object/bbox/ymin'])
            ymaxs = tf.sparse.to_dense(example['image/object/bbox/ymax'])
            labels = tf.sparse.to_dense(example['image/object/class/label'])
            
            boxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=1)
            
            return image, {'boxes': boxes, 'labels': labels}

        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

# Usage example:
# detector = ObjectDetector('path/to/saved_model', 'path/to/label_map.pbtxt', 'path/to/pipeline.config')
# image = ... # Load your image here
# detections = detector.detect_objects(image)
# detector.visualize_detections(image, detections)
# 
# # For fine-tuning:
# train_dataset = tf.data.TFRecordDataset('path/to/train.record')
# detector.fine_tune_model(train_dataset, epochs=5)
