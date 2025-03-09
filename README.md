# OBS Camera Switcher

## Overview
The OBS Camera Switcher application automatically switches OBS scenes based on face orientation detected by two cameras. Utilizing facial landmark detection and yaw estimation, it determines the camera you are facing and switches OBS scenes accordingly.

---

## Features
- Real-time face tracking with dual-camera support.
- Automatic OBS scene switching based on detected face yaw.
- GUI for camera detection, previewing, and setting configurations.
- Customizable thresholds for switching scenes and polling intervals.
- Persistent configuration saved in `settings.json`.

## Requirements
- **Python 3.7 or higher**
- **OBS Studio** with WebSocket plugin enabled
- **dlib** face landmark

## Installation

### Clone and Setup
```bash
git clone <repo-url>
cd obs-camera-switcher
pip install -r requirements.txt
```

### Required Packages
- PyQt6
- OpenCV (opencv-python)
- dlib
- obsws-python
- numpy

## Usage
- Run the application:
```bash
python main.py
```
- Use the GUI to:
  - Detect available cameras.
  - Configure OBS scenes and camera device IDs.
  - Adjust advanced settings (e.g., polling interval, yaw threshold).
  
## Configuration
- **Scene Settings:** Define OBS scenes associated with each camera.
- **Poll Interval**: Frequency of camera checking and face detection.
- **Switch Threshold**: Duration required facing a new camera before switching OBS scenes.
- **Minimum Yaw Difference**: Minimum angle difference to trigger scene change.
- **OBS Port/Password**: Configure according to your OBS WebSocket settings.

## Troubleshooting
- Ensure `shape_predictor_68_face_landmarks.dat` is placed in the project directory.
- Confirm camera accessibility and correct IDs.
- Verify OBS WebSocket plugin is active and configured properly.
- Adjust camera positioning and lighting for optimal face detection accuracy.

## License
This project is provided under the MIT License. You are free to modify, distribute, and use the software within the license terms.

---

**Happy Streaming!**

