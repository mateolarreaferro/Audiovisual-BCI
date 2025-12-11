# camera_service.py
import threading
import time
from typing import Optional, Dict
import base64

import cv2
import mediapipe as mp
import numpy as np

from osc_sender import OSCSender


class CameraService:
    """
    Camera service with MediaPipe face tracking integration.
    Extracts facial features and sends them via OSC.
    """

    def __init__(self):
        self.camera: Optional[cv2.VideoCapture] = None
        self.camera_index = 0
        self.running = False
        self.streaming = False
        self.lock = threading.Lock()
        self.osc = OSCSender()

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None

        # Latest frame and features
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_features: Dict[str, float] = {}

        # Processing thread
        self.thread: Optional[threading.Thread] = None

        # MediaPipe landmark indices for key facial features
        self.UPPER_LIP_CENTER = 13
        self.LOWER_LIP_CENTER = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.LEFT_EYE_CENTER = 159
        self.RIGHT_EYE_CENTER = 386
        self.LEFT_EYEBROW = 70
        self.NOSE_TIP = 1
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_OUTER = 263
        self.LEFT_EYE_INNER = 133
        self.RIGHT_EYE_INNER = 362

    # ---------- Camera control ----------

    def start_camera(self, camera_index: int = 0):
        """Start camera capture"""
        if self.running:
            return

        self.camera_index = camera_index

        # Try AVFoundation backend on macOS first
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        for backend in backends:
            self.camera = cv2.VideoCapture(self.camera_index, backend)
            if self.camera.isOpened():
                break
            self.camera.release()

        if not self.camera or not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        # Initialize MediaPipe
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.running = True
        self.streaming = True

        # Start processing thread
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop_camera(self):
        """Stop camera capture"""
        if not self.running:
            return

        self.running = False
        self.streaming = False

        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        # Release resources
        if self.camera:
            self.camera.release()
            self.camera = None

        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None

    # ---------- Feature extraction ----------

    def _distance_2d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate 2D Euclidean distance"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _angle_from_points(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle in degrees from two points"""
        return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    def _extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract facial features from image using MediaPipe"""
        if not self.face_mesh:
            return {}

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]

        # Convert normalized landmarks to pixel coordinates
        def get_point(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        # Compute mouth openness
        upper_lip = get_point(self.UPPER_LIP_CENTER)
        lower_lip = get_point(self.LOWER_LIP_CENTER)
        mouth_left = get_point(self.MOUTH_LEFT)
        mouth_right = get_point(self.MOUTH_RIGHT)

        mouth_height = self._distance_2d(upper_lip, lower_lip)
        mouth_width = self._distance_2d(mouth_left, mouth_right)
        mouth_openness = mouth_height / max(mouth_width, 1.0)

        # Compute eyebrow raise
        left_eye = get_point(self.LEFT_EYE_CENTER)
        left_brow = get_point(self.LEFT_EYEBROW)
        right_eye = get_point(self.RIGHT_EYE_CENTER)

        eye_height = abs(left_eye[1] - right_eye[1]) + 20
        brow_displacement = (left_eye[1] - left_brow[1]) / eye_height

        # Compute head yaw (horizontal rotation)
        left_eye_outer = get_point(self.LEFT_EYE_OUTER)
        right_eye_outer = get_point(self.RIGHT_EYE_OUTER)
        left_eye_inner = get_point(self.LEFT_EYE_INNER)
        right_eye_inner = get_point(self.RIGHT_EYE_INNER)

        left_eye_width = self._distance_2d(left_eye_outer, left_eye_inner)
        right_eye_width = self._distance_2d(right_eye_outer, right_eye_inner)
        yaw_ratio = (left_eye_width - right_eye_width) / (left_eye_width + right_eye_width)
        head_yaw = yaw_ratio * 0.5

        # Compute head roll (tilt)
        eye_angle = self._angle_from_points(left_eye_outer, right_eye_outer)
        head_roll = eye_angle

        # Compute smile curvature
        nose_tip = get_point(self.NOSE_TIP)
        mouth_center = (mouth_left + mouth_right) / 2
        mouth_curve = (mouth_center[1] - nose_tip[1]) / max(mouth_width, 1.0)
        smile_curvature = 1.0 / (1.0 + np.exp(-5 * (mouth_curve - 0.3)))

        return {
            'mouth_openness': float(mouth_openness),
            'brow_raise': float(brow_displacement),
            'head_yaw': float(head_yaw),
            'head_roll': float(head_roll),
            'smile_curvature': float(smile_curvature),
        }

    # ---------- Processing loop ----------

    def _process_loop(self):
        """Main processing loop running in separate thread"""
        while self.running:
            if not self.camera:
                time.sleep(0.1)
                continue

            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)

            # Extract features
            features = self._extract_features(frame)

            # Store latest data
            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_features = features

            # Send OSC if enabled
            if features:
                self.osc_push_features(features)

            # Limit frame rate
            time.sleep(1/30)

    # ---------- Data access ----------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest camera frame"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        """Get the latest camera frame as JPEG bytes"""
        frame = self.get_latest_frame()
        if frame is None:
            return None

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return None

        return buffer.tobytes()

    def get_latest_frame_base64(self) -> Optional[str]:
        """Get the latest camera frame as base64-encoded JPEG"""
        jpeg_bytes = self.get_latest_frame_jpeg()
        if jpeg_bytes is None:
            return None

        return base64.b64encode(jpeg_bytes).decode('utf-8')

    def get_latest_features(self) -> Dict[str, float]:
        """Get the latest extracted features"""
        with self.lock:
            return self.latest_features.copy()

    # ---------- OSC integration ----------

    def configure_osc(self, ip: str, port: int, enabled: bool):
        """Configure OSC output"""
        # For camera, we only send CV data, not raw/bands
        self.osc.configure(ip, port, enabled, send_raw=False, send_bands=False)

    def osc_push_features(self, features: Dict[str, float]):
        """Send facial features via OSC in FaceSynth-compatible format"""
        if not self.osc._ensure_client():
            return

        try:
            # Map feature names to FaceSynth format
            feature_mapping = {
                'mouth_openness': 'mouth',
                'brow_raise': 'brow',
                'head_yaw': 'yaw',
                'head_roll': 'roll',
                'smile_curvature': 'smile'
            }

            # Send each feature as a separate OSC message in FaceSynth format
            for feature_name, value in features.items():
                faceSynth_name = feature_mapping.get(feature_name, feature_name)
                # Send to both /faceSynth and /cv/face for compatibility
                self.osc.client.send_message(f"/faceSynth/{faceSynth_name}", value)
                self.osc.client.send_message(f"/cv/face/{feature_name}", value)
        except Exception as e:
            print(f"Failed to send CV features via OSC: {e}")
