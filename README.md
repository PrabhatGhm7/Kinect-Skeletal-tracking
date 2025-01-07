# Kinect Skeletal tracking

## Description

The **Kinect Skeletal tracking** is a real-time interactive Python application that uses a combination of OpenCV, MediaPipe, and Pygame to detect and animate a glowing skeleton with face landmarks. This application captures the video feed from a webcam, detects human pose landmarks and facial landmarks, and visualizes them with a glowing effect in an animated skeleton form on the screen.

## Features
- **Pose Detection**: Uses MediaPipe to detect the human body pose and draw key body points and connections (e.g., head, shoulders, elbows, knees, ankles).
- **Face Detection**: Uses MediaPipe FaceMesh to detect key facial features (e.g., eyes, eyebrows, mouth) and animate them with glowing effects.
- **Glow Effect**: The skeleton and face landmarks are drawn with a glowing effect that intensifies as the radius increases.
- **Real-time Processing**: The skeleton is drawn in real time, with a smooth update rate of 30 FPS.
- **Interactive**: The webcam feed is mirrored, providing a real-time view of the user along with their glowing skeleton and face landmarks.

## Requirements

To run this project, you'll need the following Python libraries:

- `opencv-python`
- `mediapipe`
- `pygame`
- `numpy`

You can install the required libraries using `pip`:

```bash
pip install opencv-python mediapipe pygame numpy
