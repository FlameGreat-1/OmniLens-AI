import cv2
import numpy as np
import dlib
from imutils import face_utils
import logging

class FaceDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
            self.age_net = cv2.dnn.readNetFromCaffe(
                "deploy_age.prototxt", "age_net.caffemodel")
            self.gender_net = cv2.dnn.readNetFromCaffe(
                "deploy_gender.prototxt", "gender_net.caffemodel")
        except Exception as e:
            self.logger.error(f"Error initializing FaceDetector: {str(e)}")
            raise

    def detect_faces(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 1)
            return faces
        except Exception as e:
            self.logger.error(f"Error in detect_faces: {str(e)}")
            return []

    def get_landmarks(self, image, face):
        try:
            shape = self.landmark_predictor(image, face)
            return face_utils.shape_to_np(shape)
        except Exception as e:
            self.logger.error(f"Error in get_landmarks: {str(e)}")
            return None

    def recognize_face(self, image, face):
        try:
            shape = self.landmark_predictor(image, face)
            face_descriptor = self.face_recognizer.compute_face_descriptor(image, shape)
            return np.array(face_descriptor)
        except Exception as e:
            self.logger.error(f"Error in recognize_face: {str(e)}")
            return None

    def estimate_age_gender(self, face_image):
        try:
            blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = "Male" if gender_preds[0][0] < gender_preds[0][1] else "Female"
            
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = int(age_preds[0].argmax())
            
            return age, gender
        except Exception as e:
            self.logger.error(f"Error in estimate_age_gender: {str(e)}")
            return None, None

    def detect_emotions(self, face_image):
        # This is a placeholder. In a real-world application, you would use a pre-trained
        # emotion detection model here.
        return "Neutral"

    def align_face(self, image, face):
        try:
            landmarks = self.get_landmarks(image, face)
            if landmarks is None:
                return None

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            left_eye_center = left_eye.mean(axis=0).astype("int")
            right_eye_center = right_eye.mean(axis=0).astype("int")

            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                           (left_eye_center[1] + right_eye_center[1]) // 2)

            M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
            aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

            return aligned_face
        except Exception as e:
            self.logger.error(f"Error in align_face: {str(e)}")
            return None

        def process_image(self, image):
        try:
            faces = self.detect_faces(image)
            results = []
            for face in faces:
                face_data = {}
                face_data['bbox'] = (face.left(), face.top(), face.right(), face.bottom())
                
                aligned_face = self.align_face(image, face)
                if aligned_face is not None:
                    face_image = aligned_face[face.top():face.bottom(), face.left():face.right()]
                    
                    face_data['landmarks'] = self.get_landmarks(image, face)
                    face_data['face_encoding'] = self.recognize_face(image, face)
                    face_data['age'], face_data['gender'] = self.estimate_age_gender(face_image)
                    face_data['emotion'] = self.detect_emotions(face_image)
                
                results.append(face_data)
            
            return results
        except Exception as e:
            self.logger.error(f"Error in process_image: {str(e)}")
            return []

    def track_faces(self, video_path, output_path):
        try:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            face_trackers = {}
            face_ids = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                active_face_ids = []

                # Update existing trackers
                for fid in list(face_trackers.keys()):
                    tracking_quality = face_trackers[fid].update(frame)
                    if tracking_quality < 7:
                        del face_trackers[fid]
                    else:
                        active_face_ids.append(fid)

                # Detect new faces
                if len(face_trackers) < 10:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector(gray, 1)
                    for face in faces:
                        x = face.left()
                        y = face.top()
                        w = face.right() - face.left()
                        h = face.bottom() - face.top()

                        matched_fid = None
                        for fid in face_trackers.keys():
                            tracked_position = face_trackers[fid].get_position()
                            t_x = int(tracked_position.left())
                            t_y = int(tracked_position.top())
                            t_w = int(tracked_position.width())
                            t_h = int(tracked_position.height())

                            if ((t_x <= x <= (t_x + t_w)) and
                                (t_y <= y <= (t_y + t_h)) and
                                (x <= t_x <= (x + w)) and
                                (y <= t_y <= (y + h))):
                                matched_fid = fid

                        if matched_fid is None:
                            tracker = dlib.correlation_tracker()
                            tracker.start_track(rgb_frame, dlib.rectangle(x, y, x+w, y+h))
                            face_trackers[face_ids] = tracker
                            active_face_ids.append(face_ids)
                            face_ids += 1

                # Draw rectangles around tracked faces
                for fid in active_face_ids:
                    tracked_position = face_trackers[fid].get_position()
                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())
                    cv2.rectangle(frame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face {fid}", (t_x, t_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"Error in track_faces: {str(e)}")

    def compare_faces(self, face_encoding1, face_encoding2, tolerance=0.6):
        try:
            distance = np.linalg.norm(face_encoding1 - face_encoding2)
            return distance <= tolerance
        except Exception as e:
            self.logger.error(f"Error in compare_faces: {str(e)}")
            return False

    def find_facial_features(self, landmarks):
        try:
            facial_features = {}
            facial_features['jaw'] = landmarks[0:17]
            facial_features['right_eyebrow'] = landmarks[17:22]
            facial_features['left_eyebrow'] = landmarks[22:27]
            facial_features['nose_bridge'] = landmarks[27:31]
            facial_features['nose_tip'] = landmarks[31:36]
            facial_features['right_eye'] = landmarks[36:42]
            facial_features['left_eye'] = landmarks[42:48]
            facial_features['outer_lip'] = landmarks[48:60]
            facial_features['inner_lip'] = landmarks[60:68]
            return facial_features
        except Exception as e:
            self.logger.error(f"Error in find_facial_features: {str(e)}")
            return None

    def analyze_facial_symmetry(self, landmarks):
        try:
            left_eye = np.mean(landmarks[42:48], axis=0)
            right_eye = np.mean(landmarks[36:42], axis=0)
            nose_tip = landmarks[33]
            left_mouth = landmarks[48]
            right_mouth = landmarks[54]

            eye_distance = np.linalg.norm(left_eye - right_eye)
            left_eye_to_nose = np.linalg.norm(left_eye - nose_tip)
            right_eye_to_nose = np.linalg.norm(right_eye - nose_tip)
            left_mouth_to_nose = np.linalg.norm(left_mouth - nose_tip)
            right_mouth_to_nose = np.linalg.norm(right_mouth - nose_tip)

            symmetry_score = 1 - (abs(left_eye_to_nose - right_eye_to_nose) / eye_distance +
                                  abs(left_mouth_to_nose - right_mouth_to_nose) / eye_distance) / 2

            return symmetry_score
        except Exception as e:
            self.logger.error(f"Error in analyze_facial_symmetry: {str(e)}")
            return None

    def estimate_head_pose(self, image, face):
        try:
            landmarks = self.get_landmarks(image, face)
            if landmarks is None:
                return None

            image_points = np.array([
                landmarks[30],     # Nose tip
                landmarks[8],      # Chin
                landmarks[36],     # Left eye left corner
                landmarks[45],     # Right eye right corner
                landmarks[48],     # Left Mouth corner
                landmarks[54]      # Right mouth corner
            ], dtype="double")

            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])

            size = image.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype = "double"
            )

            dist_coeffs = np.zeros((4,1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            return (rotation_vector, translation_vector)
        except Exception as e:
            self.logger.error(f"Error in estimate_head_pose: {str(e)}")
            return None
