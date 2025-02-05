# Tiny Proctor - Real-time Proctoring System

## Overview

Tiny Proctor is a lightweight, real-time proctoring system that uses computer vision to detect suspicious behavior during online examinations. It monitors both head pose and eye gaze to identify potential cheating attempts.

## Features

- Real-time face detection
- Head pose estimation (pitch, yaw, roll angles)
- Eye gaze tracking
- Suspicious behavior detection
- Visual feedback and warnings
- Debug information display

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- SciPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/tiny-proctor.git
cd tiny-proctor
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the face detection model:

```bash
# Create models directory
mkdir models
# Download YuNet face detection model
wget https://github.com/ShiqiYu/libfacedetection.train/raw/master/tasks/task1/onnx/yunet_120x160.onnx -O models/face_detection_yunet_2023mar.onnx
```

## Usage

Run the main script:

```bash
python src/main.py
```

### Controls

- Press 'q' to quit the application

## Understanding the Output

The system displays several metrics:

### Head Pose Information

- Pitch: Vertical head rotation (up/down)
- Yaw: Horizontal head rotation (left/right)
- Roll: Sideways head tilt

### Eye Gaze Information

- Vertical Ratio: Eye position up/down
- Horizontal Ratio: Eye position left/right
- Status indicators for different gaze directions

### Warning System

The system will display a warning when it detects:

- Looking down for extended periods
- Frequent left/right head movements
- Suspicious eye movements
- Combined suspicious head and eye movements

## Calibration

For optimal performance, you may need to adjust the detection thresholds in:

```python:src/detectors/eye_gaze_detector.py
startLine: 29
endLine: 44
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for face mesh detection
- OpenCV for image processing
- YuNet for face detection model
