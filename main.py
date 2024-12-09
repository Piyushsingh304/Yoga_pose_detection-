import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

class YogaPoseDetector:
    def __init__(self, model_path='pose_classifier.pkl'):
        # Initialize MediaPipe Pose and Drawing modules
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load the trained model
        try:
            self.clf = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found. Please train the model first.")
            self.clf = None

        # Define more comprehensive ideal angles for poses
        self.ideal_angles = {
            "chair": {
                "left_knee_hip_ankle": 90,
                "right_knee_hip_ankle": 90,
                "left_hip_shoulder_vertical": 90,
                "right_hip_shoulder_vertical": 90
            },
            "tree": {
                "left_knee_hip_angle": 45,
                "right_hip_balance": 90
            },
            "warrior": {
                "left_knee_hip_ankle": 90,
                "right_hip_shoulder_vertical": 180,
                "hip_rotation": 45
            }
        }

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        return angle if angle <= 180 else 360 - angle

    def extract_keypoints(self, landmarks):
        """Extract and normalize keypoints."""
        # Extract x and y coordinates for all landmarks
        keypoints = np.array([[lm.x, lm.y] for lm in landmarks])
        
        # Flatten and normalize
        keypoints_flat = keypoints.flatten()
        
        # Ensure consistent feature vector length
        if len(keypoints_flat) > 34:
            keypoints_flat = keypoints_flat[:34]
        elif len(keypoints_flat) < 34:
            keypoints_flat = np.pad(keypoints_flat, (0, 34 - len(keypoints_flat)), 'constant')
        
        return keypoints_flat / np.max(np.abs(keypoints_flat))

    def get_pose_feedback(self, landmarks, predicted_pose):
        """Generate corrective feedback for detected pose."""
        feedback = []
        
        if predicted_pose not in self.ideal_angles:
            return feedback

        # Specific landmark mappings
        lm = self.mp_pose.PoseLandmark
        landmark_map = {
            "left_knee_hip_ankle": (lm.LEFT_HIP, lm.LEFT_KNEE, lm.LEFT_ANKLE),
            "right_knee_hip_ankle": (lm.RIGHT_HIP, lm.RIGHT_KNEE, lm.RIGHT_ANKLE),
            "left_hip_shoulder_vertical": (lm.LEFT_SHOULDER, lm.LEFT_HIP, lm.LEFT_KNEE),
            "right_hip_shoulder_vertical": (lm.RIGHT_SHOULDER, lm.RIGHT_HIP, lm.RIGHT_KNEE),
            "left_knee_hip_angle": (lm.LEFT_ANKLE, lm.LEFT_HIP, lm.LEFT_SHOULDER),
            "right_hip_balance": (lm.RIGHT_HIP, lm.RIGHT_KNEE, lm.RIGHT_ANKLE),
            "hip_rotation": (lm.LEFT_HIP, lm.RIGHT_HIP, lm.LEFT_SHOULDER)
        }

        for key, ideal_angle in self.ideal_angles[predicted_pose].items():
            if key in landmark_map:
                points = [landmarks[point] for point in landmark_map[key]]
                try:
                    actual_angle = self.calculate_angle(*points)
                    if abs(actual_angle - ideal_angle) > 15:  # More lenient threshold
                        feedback.append(f"Adjust {key}: Current {actual_angle:.1f}°, Ideal {ideal_angle}°")
                except Exception as e:
                    print(f"Error calculating angle for {key}: {e}")

        return feedback

    def detect_and_correct(self):
        """Main detection and correction method."""
        cap = cv2.VideoCapture(0)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Prepare image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                    # Extract and prepare landmarks
                    landmarks = results.pose_landmarks.landmark
                    input_features = self.extract_keypoints(landmarks)

                    # Predict pose
                    if self.clf is not None:
                        pose_prediction = self.clf.predict([input_features])[0]
                        
                        # Display predicted pose
                        cv2.putText(image, f"Pose: {pose_prediction}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Get corrective feedback
                        feedback = self.get_pose_feedback(landmarks, pose_prediction)
                        
                        # Display feedback
                        for i, correction in enumerate(feedback):
                            cv2.putText(image, correction, (50, 100 + i * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Display the frame
                cv2.imshow('Yoga Pose Detection & Correction', image)

                # Exit on 'q' key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = YogaPoseDetector()
    detector.detect_and_correct()

if __name__ == "__main__":
    main()