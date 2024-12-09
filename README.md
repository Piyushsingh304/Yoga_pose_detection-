
# Real-Time Yoga Pose Detection and Correction 

## Project Overview
Overview
This project is an AI-powered solution designed to enhance the yoga experience by providing real-time feedback for pose correction and alignment. Leveraging OpenCV, MediaPipe, and machine learning, the application identifies yoga poses and offers personalized suggestions for improving form and accuracy.

The solution is composed of two main components:
1. A Python-based backend for pose detection, alignment checking, and feedback generation.
2. A React-based frontend for real-time webcam capture, pose visualization, and user interaction.

This project is inspired by the paper "Real-Time Yoga Pose Detection Using OpenCV and MediaPipe" and integrates modern AI and computer vision techniques to deliver a seamless user experience.

## Features
Python Backend
- Pose Detection: Utilizes MediaPipe's Pose module to detect body landmarks.
- Angle Calculations: Computes angles between joints to assess pose alignment.
- Pose Classification: A machine learning model (trained using scikit-learn) predicts the current yoga pose based on landmark keypoints.
- Feedback Mechanism: Provides corrective guidance based on deviations from predefined ideal pose angles.

React Frontend
- Webcam Integration: Enables users to view themselves while performing yoga.
- Real-Time Detection: Processes webcam frames and sends them to the backend for analysis.
- Feedback Display: Shows detected pose and corrective suggestions in an intuitive interface.
- Continuous Mode: Option for automatic pose detection every few seconds.


Backend Setup


1. Clone the repository:

        -git clone https://github.com/your-repo/yoga-pose-detector.git
        cd yoga-pose-detector


2. Ensure the trained pose classifier model (pose_classifier.pkl) is in the project directory.
3. Run the Python server:

        python app.py

Frontend Setup:

1. Navigate to the frontend directory:

        cd frontend

2. Install dependencies:

        npm install

3. Start the React development server:

        npm start


## Usage
1. Launch the backend server and frontend interface.
2. Use the "Start Webcam" button in the frontend to begin pose detection.
3. Perform a yoga pose in front of the webcam.
4. View the detected pose and corrective feedback in the interface.
5. Optionally, enable Continuous Mode for hands-free operation.

## Technology Stack
- Python: Backend processing and pose detection.
- MediaPipe: Landmark detection and pose estimation.
- scikit-learn: Machine learning-based pose classification.
- OpenCV: Frame capture and processing.
- React: Frontend user interface.

## Key Components
Backend
-YogaPoseDetector: Main class for pose detection and feedback generation.
- calculate_angle: Computes joint angles using trigonometry.
- extract_keypoints: Normalizes landmark data for classification.

Frontend
- YogaPoseDetector Component: Manages webcam feed and user interactions.
- Feedback Display: Dynamically renders corrective guidance based on backend results.

## Contributions
1. Pose Detection: Inspired by "Real-Time Yoga Pose Detection Using OpenCV and MediaPipe."
2.  Feedback System: Developed to guide users in improving their form based on joint alignment.
3. Continuous Detection: Added for convenience during longer yoga sessions.

## Future Enhancements
- Pose Customization: Allow users to define custom ideal poses.
- Performance Tracking: Track progress over time with analytics.
- Voice Feedback: Provide auditory instructions alongside visual cues.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

##References
- "Real-Time Yoga Pose Detection Using OpenCV and MediaPipe"
- MediaPipe Documentation: https://google.github.io/mediapipe/
- OpenCV Documentation: https://docs.opencv.org/

## Acknowledgments
Special thanks to the creators of MediaPipe and OpenCV for their powerful tools enabling this project.