import sys
import time
import subprocess
import cv2
import dlib
import numpy as np
import json
import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QDoubleSpinBox, QSpinBox, QMessageBox,
    QGridLayout, QGroupBox, QComboBox, QScrollArea
)


##############################################################################
# Camera Worker Thread (QThread)
##############################################################################
class CameraWorker(QThread):
    """
    Worker thread that opens two cameras and monitors face yaw.
    Switches scenes in OBS based on user-selected parameters.
    """
    # Signal to update status in GUI
    status_update_signal = pyqtSignal(str)

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

        # obs-cli args
        self.obs_cli_args = ["obs-cli"]  # Adjust if needed

        # DLIB initialization
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(self.predictor_path)

        # 3D model points
        self.model_points_3D = np.array([
            (0.0,      0.0,       0.0),    # Nose tip
            (0.0,     -330.0,    -65.0),   # Chin
            (-165.0,   170.0,    -135.0),  # Left corner of left eye
            (165.0,    170.0,    -135.0),  # Right corner of the right eye
            (-150.0,  -150.0,    -125.0),  # Left corner of mouth
            (150.0,   -150.0,    -125.0)   # Right corner of mouth
        ], dtype=np.float32)

        # Downscale factor for faster face detection
        self.downscale_factor = 2

        # Variables for scene switching logic
        self.last_facing = None
        self.camera_facing = None
        self.yaw_difference = None
        self.camera_switch_counter = 0
        self.camera_switch_counter_threshold = 0

        # We create the captures in run()
        self.cap0 = None
        self.cap1 = None

    def run(self):
        """
        Main loop. Opens the cameras with the selected device IDs
        and runs until self._running is False.
        """
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
            self.status_update_signal.emit(msg)

        while self._running:
            if not self.cap0 or not self.cap1:
                time.sleep(1)
                continue

            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()

            if not ret0 or not ret1:
                err_msg = "Error: Could not read from one or both cameras."
                print(err_msg)
                self.status_update_signal.emit(err_msg)
                time.sleep(1)
                continue

            # Downsize frames
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

            yaw0 = result0[0] if result0 else None
            yaw1 = result1[0] if result1 else None

            # Decide which camera the subject is facing
            camera_facing = None
            if yaw0 is not None and yaw1 is not None:
                self.yaw_difference = abs(yaw0 - yaw1)
                if abs(yaw0) < abs(yaw1):
                    camera_facing = "Camera 0"
                else:
                    camera_facing = "Camera 1"
            elif yaw0 is not None:
                camera_facing = "Camera 0"
            elif yaw1 is not None:
                camera_facing = "Camera 1"

            # Scene switching logic
            if camera_facing and camera_facing != self.last_facing and self.yaw_difference and self.yaw_difference >= self.minimum_yaw_diff_to_switch:
                self.camera_switch_counter += 1
                if self.camera_switch_counter >= self.camera_switch_counter_threshold:
                    self.camera_switch_counter = 0
                    self.switch_obs_scene(camera_facing)
                    self.last_facing = camera_facing
            else:
                self.camera_switch_counter = 0

            # Update status
            if camera_facing:
                status = f"Facing: {camera_facing}, Yaw Diff: {self.yaw_difference}"
            else:
                status = "No face or not enough difference"
            self.status_update_signal.emit(status)

            time.sleep(self.poll_interval)

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
        Switch the active OBS scene using obs-cli.
        """
        scene_name = None
        if camera_facing == "Camera 0":
            scene_name = self.scene_camera_0
        elif camera_facing == "Camera 1":
            scene_name = self.scene_camera_1

        if scene_name:
            cmd = self.obs_cli_args + ["scene", "switch", scene_name]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to switch scene to {scene_name}: {e}")
            except FileNotFoundError:
                print("[ERROR] obs-cli not found. Make sure it’s installed and in your PATH.")

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
        param_layout = QGridLayout()

        # -- Camera 0 row --
        label_cam0 = QLabel("Camera 0:")
        self.edit_scene0 = QLineEdit(self.worker.scene_camera_0)
        self.combo_cam0_id = QComboBox()
        # We'll populate the combo box after we detect cameras
        param_layout.addWidget(label_cam0,        0, 0)
        param_layout.addWidget(self.edit_scene0,  0, 1)
        param_layout.addWidget(self.combo_cam0_id,0, 2)

        # -- Camera 1 row --
        label_cam1 = QLabel("Camera 1:")
        self.edit_scene1 = QLineEdit(self.worker.scene_camera_1)
        self.combo_cam1_id = QComboBox()
        param_layout.addWidget(label_cam1,        1, 0)
        param_layout.addWidget(self.edit_scene1,  1, 1)
        param_layout.addWidget(self.combo_cam1_id,1, 2)

        # Poll Interval
        label_poll = QLabel("Poll Interval (s):")
        self.spin_poll = QDoubleSpinBox()
        self.spin_poll.setValue(self.worker.poll_interval)
        self.spin_poll.setRange(0.01, 10.0)
        self.spin_poll.setSingleStep(0.05)
        param_layout.addWidget(label_poll, 2, 0)
        param_layout.addWidget(self.spin_poll, 2, 1)

        # Switch threshold
        label_thresh = QLabel("Switch Threshold (s):")
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setValue(self.worker.camera_switch_threshold)
        self.spin_thresh.setRange(0.0, 10.0)
        self.spin_thresh.setSingleStep(0.5)
        param_layout.addWidget(label_thresh, 3, 0)
        param_layout.addWidget(self.spin_thresh, 3, 1)

        # Minimum yaw difference
        label_yawdiff = QLabel("Min Yaw Diff (deg):")
        self.spin_yawdiff = QSpinBox()
        self.spin_yawdiff.setValue(self.worker.minimum_yaw_diff_to_switch)
        self.spin_yawdiff.setRange(0, 90)
        param_layout.addWidget(label_yawdiff, 4, 0)
        param_layout.addWidget(self.spin_yawdiff, 4, 1)

        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        # ===================================
        # 3) Start/Stop Buttons + Status
        # ===================================
        button_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_worker)
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)

        self.status_label = QLabel("Status: Idle")

        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)


    @pyqtSlot()
    def detect_cameras(self):
        """
        Attempt to open camera IDs 0..10, capture a single frame,
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

        # We define a range of potential devices to test
        found_devices = []
        device_id = 0
        while True:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                time.sleep(0.2)  # Let the camera warm up
                ret, frame = cap.read()
                if ret:
                    found_devices.append(device_id)
                    # Convert this frame to a QPixmap for preview
                    preview_label = self.create_preview_label(frame, device_id)
                    self.preview_layout.addWidget(preview_label)
                    self.detect_group.setFixedHeight(min(2,len(found_devices)) * 180)
                cap.release()
            else:
                break
            device_id += 1

        if found_devices:
            # Fill combo boxes with found devices
            for dev_id in found_devices:
                self.combo_cam0_id.addItem(str(dev_id), dev_id)
                self.combo_cam1_id.addItem(str(dev_id), dev_id)
        else:
            # No devices found
            no_cam_label = QLabel("No cameras found.")
            self.preview_layout.addWidget(no_cam_label)

    def create_preview_label(self, frame, device_id):
        """
        Convert an OpenCV frame to a QPixmap and embed in a QLabel
        with some text describing the device ID.
        """
        # Convert BGR to RGB
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

        # Convert to QImage
        qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Create a label that shows the device ID + pixmap
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

            # Use the "data" from selected item if available
            idx0 = self.combo_cam0_id.currentIndex()
            idx1 = self.combo_cam1_id.currentIndex()

            # If there's at least one item in combo, read it, else default
            if idx0 >= 0:
                self.worker.camera0_id = self.combo_cam0_id.itemData(idx0)
            if idx1 >= 0:
                self.worker.camera1_id = self.combo_cam1_id.itemData(idx1)

            self.worker.poll_interval = self.spin_poll.value()
            self.worker.camera_switch_threshold = self.spin_thresh.value()
            self.worker.minimum_yaw_diff_to_switch = self.spin_yawdiff.value()

            self.worker.start()
            self.btn_start.setText("Apply")
            self.status_label.setText("Status: Re-started...")
        else:
            # Update worker parameters from GUI
            self.worker.scene_camera_0 = self.edit_scene0.text().strip()
            self.worker.scene_camera_1 = self.edit_scene1.text().strip()

            # Use the "data" from selected item if available
            idx0 = self.combo_cam0_id.currentIndex()
            idx1 = self.combo_cam1_id.currentIndex()

            # If there's at least one item in combo, read it, else default
            if idx0 >= 0:
                self.worker.camera0_id = self.combo_cam0_id.itemData(idx0)
            if idx1 >= 0:
                self.worker.camera1_id = self.combo_cam1_id.itemData(idx1)

            self.worker.poll_interval = self.spin_poll.value()
            self.worker.camera_switch_threshold = self.spin_thresh.value()
            self.worker.minimum_yaw_diff_to_switch = self.spin_yawdiff.value()

            self.worker.start()
            self.btn_start.setText("Apply")
            self.status_label.setText("Status: Running...")
        self.save_settings()

    @pyqtSlot()
    def stop_worker(self):
        """
        Stop the camera processing thread.
        """
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.status_label.setText("Status: Stopped")
        self.btn_start.setText("Start")

    @pyqtSlot(str)
    def on_status_update(self, msg):
        """
        Called whenever the worker emits a status update signal.
        """
        self.status_label.setText(f"Status: {msg}")

    def closeEvent(self, event):
        """
        If the user closes the window, ensure the worker is stopped.
        """
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.save_settings()
        event.accept()

    def load_settings(self):
        """Load settings from settings.json if available."""
        if os.path.exists("settings.json"):
            with open("settings.json", "r") as f:
                data = json.load(f)
            self.edit_scene0.setText(data.get("scene_camera_0", self.worker.scene_camera_0))
            self.edit_scene1.setText(data.get("scene_camera_1", self.worker.scene_camera_1))
            self.spin_poll.setValue(data.get("poll_interval", self.worker.poll_interval))
            self.spin_thresh.setValue(data.get("camera_switch_threshold", self.worker.camera_switch_threshold))
            self.spin_yawdiff.setValue(data.get("minimum_yaw_diff_to_switch", self.worker.minimum_yaw_diff_to_switch))
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
        """Save current settings to settings.json."""
        data = {
            "scene_camera_0": self.edit_scene0.text().strip(),
            "scene_camera_1": self.edit_scene1.text().strip(),
            "poll_interval": self.spin_poll.value(),
            "camera_switch_threshold": self.spin_thresh.value(),
            "minimum_yaw_diff_to_switch": self.spin_yawdiff.value(),
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