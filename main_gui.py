import sys
import time
import cv2
import dlib
import numpy as np
import json
import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QDoubleSpinBox, QSpinBox, QGroupBox, QComboBox,
    QScrollArea
)

import obsws_python as obsws

##############################################################################
# Camera Worker Thread (QThread)
##############################################################################
class CameraWorker(QThread):
    """
    Worker thread that opens two cameras and monitors face yaw.
    Switches scenes in OBS based on user-selected parameters.
    """
    # We emit a dictionary with comprehensive status info
    status_update_signal = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Thread control
        self._running = False

        # Default parameters
        self.scene_camera_0 = "Scene 0"
        self.scene_camera_1 = "Scene 1"

        # Device IDs for camera0, camera1
        self.camera0_id = 0
        self.camera1_id = 1

        # Timing/angle thresholds
        self.poll_interval = 0.1
        self.camera_switch_threshold = 0.5
        self.minimum_yaw_diff_to_switch = 50

        # OBS settings
        self.obs_port = 4455
        self.obs_password = ""

        # DLIB initialization
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(self.predictor_path)

        # 3D model points
        self.model_points_3D = np.array([
            (0.0,      0.0,       0.0),    # Nose tip
            (0.0,     -330.0,    -65.0),   # Chin
            (-165.0,   170.0,    -135.0),  # Left eye corner
            (165.0,    170.0,    -135.0),  # Right eye corner
            (-150.0,  -150.0,    -125.0),  # Left corner of mouth
            (150.0,   -150.0,    -125.0)   # Right corner of mouth
        ], dtype=np.float32)

        # Downscale factor for faster face detection
        self.downscale_factor = 2

        # Variables for scene switching logic
        self.last_facing = None
        self.obs_facing = None  # Tracks the most recently set OBS scene
        self.yaw_difference = None
        self.camera_switch_counter = 0
        self.camera_switch_counter_threshold = 0

        # Captures
        self.cap0 = None
        self.cap1 = None

        # Store the most recent yaws for each camera
        self.yaw0 = None
        self.yaw1 = None

    def run(self):
        self._running = True

        # Convert switch threshold (seconds) to iteration count
        self.camera_switch_counter_threshold = int(
            self.camera_switch_threshold / self.poll_interval
        )

        # Open cameras
        self.cap0 = cv2.VideoCapture(self.camera0_id)
        self.cap1 = cv2.VideoCapture(self.camera1_id)

        if not self.cap0.isOpened() or not self.cap1.isOpened():
            msg = "Error: Could not open one or both cameras."
            print(msg)
            # Emit a dictionary with partial info so the GUI can update
            self.status_update_signal.emit({
                'camera_facing': None,
                'yaw0': None,
                'yaw1': None,
                'yaw_diff': None,
                'obs_facing': self.obs_facing,
                'running': self._running,
                'error': msg
            })
            return

        while self._running:
            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()

            if not ret0 or not ret1:
                err_msg = "Error: Could not read from one or both cameras."
                print(err_msg)
                self.status_update_signal.emit({
                    'camera_facing': None,
                    'yaw0': None,
                    'yaw1': None,
                    'yaw_diff': None,
                    'obs_facing': self.obs_facing,
                    'running': self._running,
                    'error': err_msg
                })
                time.sleep(1)
                continue

            # Downsize frames for faster face detection
            frame0_small = cv2.resize(
                frame0,
                (frame0.shape[1] // self.downscale_factor,
                 frame0.shape[0] // self.downscale_factor)
            )
            frame1_small = cv2.resize(
                frame1,
                (frame1.shape[1] // self.downscale_factor,
                 frame1.shape[0] // self.downscale_factor)
            )

            # Estimate yaw
            result0 = self.estimate_yaw(frame0_small)
            result1 = self.estimate_yaw(frame1_small)

            self.yaw0 = result0[0] if result0 else None
            self.yaw1 = result1[0] if result1 else None

            # Decide which camera the subject is facing
            camera_facing = None
            if self.yaw0 is not None and self.yaw1 is not None:
                self.yaw_difference = abs(self.yaw0 - self.yaw1)
                # Closer to zero means more "centered" to that camera
                if abs(self.yaw0) < abs(self.yaw1):
                    camera_facing = "Camera 0"
                else:
                    camera_facing = "Camera 1"
            elif self.yaw0 is not None:
                camera_facing = "Camera 0"
                self.yaw_difference = None
            elif self.yaw1 is not None:
                camera_facing = "Camera 1"
                self.yaw_difference = None
            else:
                self.yaw_difference = None

            # Scene switching logic
            if (camera_facing and
                camera_facing != self.last_facing and
                (self.yaw_difference is None or
                (self.yaw_difference is not None and
                self.yaw_difference >= self.minimum_yaw_diff_to_switch))):

                self.camera_switch_counter += 1
                if self.camera_switch_counter >= self.camera_switch_counter_threshold:
                    self.camera_switch_counter = 0
                    self.switch_obs_scene(camera_facing)
                    self.last_facing = camera_facing
            else:
                self.camera_switch_counter = 0

            # Emit comprehensive status info every loop
            self.status_update_signal.emit({
                'camera_facing': camera_facing,
                'yaw0': self.yaw0,
                'yaw1': self.yaw1,
                'yaw_diff': self.yaw_difference,
                'obs_facing': self.obs_facing,
                'running': self._running,
                'error': None
            })

            time.sleep(self.poll_interval)

        # Cleanup
        if self.cap0:
            self.cap0.release()
        if self.cap1:
            self.cap1.release()

    def stop(self):
        """
        Signal the thread to stop gracefully.
        """
        self._running = False

    def switch_obs_scene(self, camera_facing):
        """
        Switch the active OBS scene using obsws.
        """
        scene_name = None
        if camera_facing == "Camera 0":
            scene_name = self.scene_camera_0
        elif camera_facing == "Camera 1":
            scene_name = self.scene_camera_1

        if scene_name:
            try:
                self.obs_client.set_current_program_scene(scene_name)
                # Track the most recently set OBS scene
                self.obs_facing = camera_facing
            except Exception as e:
                print(f"[ERROR] {e}")

    def estimate_yaw(self, frame_small):
        """
        Detect a face in the (downsampled) frame, estimate its yaw (in degrees).
        Returns (yaw, (x1, y1, x2, y2)) or None if no face.
        """
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 0)

        if len(faces) == 0:
            return None

        face_rect = faces[0]
        x1, y1, x2, y2 = (face_rect.left(), face_rect.top(),
                          face_rect.right(), face_rect.bottom())

        landmarks_2D = self.get_key_landmarks(gray, face_rect)

        height, width = frame_small.shape[:2]
        focal_length = width  # approximate
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0,             center[0]],
            [0,            focal_length,  center[1]],
            [0,            0,             1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points_3D, landmarks_2D, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = eulerAngles.flatten()

        return (yaw, (x1, y1, x2, y2))

    def get_key_landmarks(self, gray, face_rect):
        """
        Return the 6 key landmarks used for PnP.
        """
        shape = self.landmark_predictor(gray, face_rect)
        landmarks_2D = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x,  shape.part(8).y),   # Chin
            (shape.part(36).x, shape.part(36).y),  # Left eye corner
            (shape.part(45).x, shape.part(45).y),  # Right eye corner
            (shape.part(48).x, shape.part(48).y),  # Left mouth corner
            (shape.part(54).x, shape.part(54).y),  # Right mouth corner
        ], dtype=np.float32)
        return landmarks_2D


##############################################################################
# Main Window / GUI
##############################################################################
class MainWindow(QMainWindow):
    """
    Main application window:
    1) 'Detect Cameras' to list available devices & show a preview frame
    2) Let user pick device IDs for Camera 0 / Camera 1 from drop-down
    3) Let user choose OBS scene names for each camera
    4) Start/Stop the worker thread with face tracking.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OBS Camera Switcher (PyQt6)")

        self.worker = CameraWorker()

        # Build the GUI
        self.init_ui()

        # Connect signals
        self.worker.status_update_signal.connect(self.on_status_update)

        # Automatically detect cameras
        self.detect_cameras()
        self.load_settings()

    def init_ui(self):
        """
        Create all widgets for the GUI.
        """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # ==========================
        # 1) Device Detection Group
        # ==========================
        self.detect_group = QGroupBox("Camera Discovery")
        detect_layout = QVBoxLayout()

        # "Refresh / Detect Cameras" button
        self.btn_detect = QPushButton("Refresh / Detect Cameras")
        self.btn_detect.clicked.connect(self.detect_cameras)
        detect_layout.addWidget(self.btn_detect)

        # Scroll area to hold camera previews
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        detect_layout.addWidget(self.scroll_area)

        # We'll fill this widget with previews
        self.preview_container = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_container)
        self.scroll_area.setWidget(self.preview_container)

        self.detect_group.setLayout(detect_layout)
        main_layout.addWidget(self.detect_group)

        # =======================================
        # 2) Parameter Group (scene names, etc.)
        # =======================================
        param_group = QGroupBox("Camera & Scene Settings")
        param_layout = QHBoxLayout()

        # Left column
        left_layout = QVBoxLayout()
        label_cam0 = QLabel("Camera 0 Scene:")
        self.edit_scene0 = QLineEdit(self.worker.scene_camera_0)
        left_layout.addWidget(label_cam0)
        left_layout.addWidget(self.edit_scene0)

        label_cam1 = QLabel("Camera 1 Scene:")
        self.edit_scene1 = QLineEdit(self.worker.scene_camera_1)
        left_layout.addWidget(label_cam1)
        left_layout.addWidget(self.edit_scene1)

        # Right column
        right_layout = QVBoxLayout()

        # Combos for camera IDs
        self.combo_cam0_id = QComboBox()
        label_cam0_id = QLabel("Device ID for Camera 0:")
        right_layout.addWidget(label_cam0_id)
        right_layout.addWidget(self.combo_cam0_id)

        self.combo_cam1_id = QComboBox()
        label_cam1_id = QLabel("Device ID for Camera 1:")
        right_layout.addWidget(label_cam1_id)
        right_layout.addWidget(self.combo_cam1_id)

        # Put them together
        param_layout.addLayout(left_layout)
        param_layout.addLayout(right_layout)
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        # =================================
        # 3) Advanced Settings Group
        # =================================
        adv_group = QGroupBox("Advanced Settings")
        adv_layout = QHBoxLayout()

        # Column 1
        col1 = QVBoxLayout()
        label_poll = QLabel("Poll Interval (s):")
        self.spin_poll = QDoubleSpinBox()
        self.spin_poll.setValue(self.worker.poll_interval)
        self.spin_poll.setRange(0.01, 10.0)
        self.spin_poll.setSingleStep(0.05)

        label_thresh = QLabel("Switch Threshold (s):")
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setValue(self.worker.camera_switch_threshold)
        self.spin_thresh.setRange(0.0, 10.0)
        self.spin_thresh.setSingleStep(0.5)

        col1.addWidget(label_poll)
        col1.addWidget(self.spin_poll)
        col1.addWidget(label_thresh)
        col1.addWidget(self.spin_thresh)

        # Column 2
        col2 = QVBoxLayout()
        label_yawdiff = QLabel("Min Yaw Diff (deg):")
        self.spin_yawdiff = QSpinBox()
        self.spin_yawdiff.setValue(self.worker.minimum_yaw_diff_to_switch)
        self.spin_yawdiff.setRange(0, 90)

        label_port = QLabel("OBS Port:")
        self.spin_port = QSpinBox()
        self.spin_port.setValue(self.worker.obs_port)
        self.spin_port.setRange(1, 65535)

        col2.addWidget(label_yawdiff)
        col2.addWidget(self.spin_yawdiff)
        col2.addWidget(label_port)
        col2.addWidget(self.spin_port)

        # Column 3
        col3 = QVBoxLayout()
        label_pass = QLabel("OBS Password:")
        self.edit_password = QLineEdit(self.worker.obs_password)
        self.edit_password.setEchoMode(QLineEdit.EchoMode.Password)

        col3.addWidget(label_pass)
        col3.addWidget(self.edit_password)

        adv_layout.addLayout(col1)
        adv_layout.addLayout(col2)
        adv_layout.addLayout(col3)
        adv_group.setLayout(adv_layout)
        main_layout.addWidget(adv_group)

        # ===================================
        # 4) Start/Stop Buttons
        # ===================================
        button_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_worker)
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)
        main_layout.addLayout(button_layout)

        # ===================================
        # 5) Status Info Labels
        # ===================================
        self.status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        # We'll create labels for each line of info
        self.lbl_status_runstop = QLabel("Status: Stopped")
        self.lbl_facing_current = QLabel("Current Facing: None")
        self.lbl_facing_obs = QLabel("Current OBS Facing: None")
        self.lbl_yaw_cam1 = QLabel("Camera 1 Yaw: N/A")
        self.lbl_yaw_cam2 = QLabel("Camera 2 Yaw: N/A")
        self.lbl_yaw_offset = QLabel("Yaw Offset: N/A")
        self.lbl_error = QLabel("")  # Only show if needed

        # Add to layout
        status_layout.addWidget(self.lbl_status_runstop)
        status_layout.addWidget(self.lbl_facing_current)
        status_layout.addWidget(self.lbl_facing_obs)
        status_layout.addWidget(self.lbl_yaw_cam1)
        status_layout.addWidget(self.lbl_yaw_cam2)
        status_layout.addWidget(self.lbl_yaw_offset)
        status_layout.addWidget(self.lbl_error)

        self.status_group.setLayout(status_layout)
        main_layout.addWidget(self.status_group)

    @pyqtSlot()
    def detect_cameras(self):
        """
        Attempt to open camera IDs sequentially, capture a single frame,
        and display them in a preview panel. Also fill the camera
        combo boxes with the IDs that worked.
        """
        # Clear out old previews
        for i in reversed(range(self.preview_layout.count())):
            widget_item = self.preview_layout.itemAt(i)
            widget = widget_item.widget()
            if widget:
                widget.setParent(None)

        # Clear combo boxes
        self.combo_cam0_id.clear()
        self.combo_cam1_id.clear()

        found_devices = []
        device_id = 0
        while True:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                time.sleep(0.2)
                ret, frame = cap.read()
                if ret:
                    found_devices.append(device_id)
                    # Create a preview
                    preview_label = self.create_preview_label(frame, device_id)
                    self.preview_layout.addWidget(preview_label)
                cap.release()
                device_id += 1
            else:
                cap.release()
                break

        if found_devices:
            for dev_id in found_devices:
                self.combo_cam0_id.addItem(str(dev_id), dev_id)
                self.combo_cam1_id.addItem(str(dev_id), dev_id)
        else:
            no_cam_label = QLabel("No cameras found.")
            self.preview_layout.addWidget(no_cam_label)

    def create_preview_label(self, frame, device_id):
        """
        Convert an OpenCV frame to a QPixmap and embed in a QLabel
        with some text describing the device ID.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Optionally resize the preview
        max_w = 320
        max_h = 100
        h, w, _ = frame_rgb.shape

        scale = min(max_w / w, max_h / h)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            h, w, _ = frame_rgb.shape

        qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        container = QWidget()
        layout = QHBoxLayout(container)

        info_label = QLabel(f"Device {device_id}")
        pic_label = QLabel()
        pic_label.setPixmap(pixmap)
        pic_label.setFixedSize(w, h)

        layout.addWidget(info_label)
        layout.addWidget(pic_label)

        return container

    @pyqtSlot()
    def start_worker(self):
        """
        Start the camera processing thread with updated parameters.
        """
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        # Update worker parameters from GUI
        self.worker.scene_camera_0 = self.edit_scene0.text().strip()
        self.worker.scene_camera_1 = self.edit_scene1.text().strip()

        idx0 = self.combo_cam0_id.currentIndex()
        idx1 = self.combo_cam1_id.currentIndex()
        if idx0 >= 0:
            self.worker.camera0_id = self.combo_cam0_id.itemData(idx0)
        if idx1 >= 0:
            self.worker.camera1_id = self.combo_cam1_id.itemData(idx1)

        self.worker.poll_interval = self.spin_poll.value()
        self.worker.camera_switch_threshold = self.spin_thresh.value()
        self.worker.minimum_yaw_diff_to_switch = self.spin_yawdiff.value()
        self.worker.obs_port = self.spin_port.value()
        self.worker.obs_password = self.edit_password.text()
        self.worker.obs_client = obsws.ReqClient(
            host="localhost", 
            port=self.worker.obs_port, 
            password=self.worker.obs_password
        )

        self.worker.start()
        self.btn_start.setText("Apply")
        self.save_settings()

    @pyqtSlot()
    def stop_worker(self):
        """
        Stop the camera processing thread.
        """
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.btn_start.setText("Start")

        # Update the label to show Stopped
        self.lbl_status_runstop.setText("Status: Stopped")

    @pyqtSlot(dict)
    def on_status_update(self, data):
        """
        Handle comprehensive status info.
        Update our dedicated status labels accordingly.
        """
        is_running = data['running']
        camera_facing = data.get('camera_facing', None)
        obs_facing = data.get('obs_facing', None)
        yaw0 = data.get('yaw0', None)
        yaw1 = data.get('yaw1', None)
        yaw_diff = data.get('yaw_diff', None)
        err_msg = data.get('error', "")

        # 1) Running/Stopped
        if is_running:
            self.lbl_status_runstop.setText("Status: Running")
        else:
            self.lbl_status_runstop.setText("Status: Stopped")

        # 2) Current Facing
        if camera_facing is None:
            self.lbl_facing_current.setText("Current Facing: No face detected")
        else:
            self.lbl_facing_current.setText(f"Current Facing: {camera_facing}")

        # 3) Current OBS Facing
        if obs_facing is None:
            self.lbl_facing_obs.setText("Current OBS Facing: None")
        else:
            self.lbl_facing_obs.setText(f"Current OBS Facing: {obs_facing}")

        # 4) Camera 1 Yaw
        #    Recall that in your earlier code, "Camera 1" is camera0 internally.
        if yaw0 is not None:
            self.lbl_yaw_cam1.setText(f"Camera 1 Yaw: {yaw0:.2f} Degrees")
        else:
            self.lbl_yaw_cam1.setText("Camera 1 Yaw: N/A")

        # 5) Camera 2 Yaw (camera1 in code)
        if yaw1 is not None:
            self.lbl_yaw_cam2.setText(f"Camera 2 Yaw: {yaw1:.2f} Degrees")
        else:
            self.lbl_yaw_cam2.setText("Camera 2 Yaw: N/A")

        # 6) Yaw Offset
        if yaw_diff is not None:
            self.lbl_yaw_offset.setText(f"Yaw Offset: {yaw_diff:.2f} Degrees")
        else:
            self.lbl_yaw_offset.setText("Yaw Offset: N/A")

        # 7) Error message (if any)
        if err_msg:
            self.lbl_error.setText(f"Error: {err_msg}")
        else:
            self.lbl_error.setText("")

    def closeEvent(self, event):
        """
        On close, ensure the worker is stopped, and save settings.
        """
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.save_settings()
        event.accept()

    def load_settings(self):
        """Load settings from a JSON file if available."""
        if os.path.exists("settings.json"):
            with open("settings.json", "r") as f:
                data = json.load(f)
            self.edit_scene0.setText(data.get("scene_camera_0", self.worker.scene_camera_0))
            self.edit_scene1.setText(data.get("scene_camera_1", self.worker.scene_camera_1))
            self.spin_poll.setValue(data.get("poll_interval", self.worker.poll_interval))
            self.spin_thresh.setValue(data.get("camera_switch_threshold", self.worker.camera_switch_threshold))
            self.spin_yawdiff.setValue(data.get("minimum_yaw_diff_to_switch", self.worker.minimum_yaw_diff_to_switch))
            self.spin_port.setValue(data.get("obs_port", self.worker.obs_port))
            self.edit_password.setText(data.get("obs_password", self.worker.obs_password))

            cam0_id = data.get("camera0_id", self.worker.camera0_id)
            cam1_id = data.get("camera1_id", self.worker.camera1_id)
            idx0 = self.combo_cam0_id.findData(cam0_id)
            if idx0 < 0:
                idx0 = self.combo_cam0_id.findData(0)
            self.combo_cam0_id.setCurrentIndex(idx0 if idx0 >= 0 else 0)

            idx1 = self.combo_cam1_id.findData(cam1_id)
            if idx1 < 0:
                idx1 = self.combo_cam1_id.findData(1)
            self.combo_cam1_id.setCurrentIndex(idx1 if idx1 >= 0 else 0)

    def save_settings(self):
        """Save current settings to a JSON file."""
        data = {
            "scene_camera_0": self.edit_scene0.text().strip(),
            "scene_camera_1": self.edit_scene1.text().strip(),
            "poll_interval": self.spin_poll.value(),
            "camera_switch_threshold": self.spin_thresh.value(),
            "minimum_yaw_diff_to_switch": self.spin_yawdiff.value(),
            "obs_port": self.spin_port.value(),
            "obs_password": self.edit_password.text(),
            "camera0_id": self.combo_cam0_id.itemData(self.combo_cam0_id.currentIndex()),
            "camera1_id": self.combo_cam1_id.itemData(self.combo_cam1_id.currentIndex())
        }
        with open("settings.json", "w") as f:
            json.dump(data, f, indent=2)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()