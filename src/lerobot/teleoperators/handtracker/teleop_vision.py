import argparse
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode

from lerobot.teleoperators.handtracker.relative_xy_mapping import RelativeXYMapper


MODEL_PATH = Path(
    os.environ.get(
        "HAND_LANDMARKER_MODEL_PATH",
        Path(__file__).with_name("models") / "hand_landmarker.task",
    )
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Missing hand landmarker model. Set HAND_LANDMARKER_MODEL_PATH or place "
        f"a model file at {MODEL_PATH}."
    )

ALLOWED_LANDMARKS = {0, 1, 2, 3, 4, 5, 6, 7, 8}
DEFAULT_XY_SCALE_M = 0.05
MAX_DELTA_FRAME_FRACTION = 0.25
DEFAULT_Z_SCALE_M = 0.08
MIN_Z_SCALE_M = 0.01
MAX_Z_SCALE_M = 0.30
MAX_DELTA_HEIGHT_FRACTION = 0.30
DEADZONE_HEIGHT_FRACTION = 0.0125
RERUN_IMAGE_EVERY_N = 3
DEADZONE_FRAME_FRACTION = 0.0125
MIN_XY_SCALE_M = 0.01
MAX_XY_SCALE_M = 0.20
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
BLACK_FRAME = np.zeros((DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, 3), dtype=np.uint8)
_RERUN_LOGGING_ENABLED = False
_RERUN_LOGGING_MODE = "off"
LOG_DIR = Path(__file__).with_name("logs")
_RUN_LOGGER: logging.Logger | None = None
_RUN_LOG_PATH: Path | None = None


def _black_frame() -> np.ndarray:
    return BLACK_FRAME.copy()


def _is_valid_frame(frame: np.ndarray | None) -> bool:
    return frame is not None and isinstance(frame, np.ndarray) and frame.size > 0


def _make_landmarker() -> hand_landmarker.HandLandmarker:
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=base_options_lib.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return hand_landmarker.HandLandmarker.create_from_options(options)


def enable_rerun_logging(enabled: bool = True, mode: str | None = "off") -> None:
    global _RERUN_LOGGING_ENABLED, _RERUN_LOGGING_MODE
    _RERUN_LOGGING_ENABLED = bool(enabled)
    if mode is not None:
        _RERUN_LOGGING_MODE = str(mode).lower()


def set_rerun_logging_mode(mode: str) -> None:
    global _RERUN_LOGGING_MODE
    if mode is None:
        return
    mode_name = str(mode).lower()
    if mode_name in {"off", "lightweight", "full"}:
        _RERUN_LOGGING_MODE = mode_name
    else:
        _RERUN_LOGGING_MODE = "off"


def _setup_run_logger() -> logging.Logger:
    global _RUN_LOGGER, _RUN_LOG_PATH

    if _RUN_LOGGER is not None:
        return _RUN_LOGGER

    shared_log_path = os.environ.get("HANDTRACKER_LOG_PATH")
    if shared_log_path:
        _RUN_LOG_PATH = Path(shared_log_path)
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _RUN_LOG_PATH = LOG_DIR / f"handtracker_{timestamp}_{os.getpid()}.log"

    logger = logging.getLogger("lerobot.teleoperators.handtracker")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not shared_log_path:
        file_handler = logging.FileHandler(_RUN_LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    _RUN_LOGGER = logger
    logger.info("Handtracker log file: %s", _RUN_LOG_PATH)
    return logger


def _rotate_frame(frame: np.ndarray, rotation_deg: int = 0) -> np.ndarray:
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _mirror_frame(frame: np.ndarray, x_mirror: bool = False, y_mirror: bool = False) -> np.ndarray:
    """Rotation is applied first; xmirror flips left/right, ymirror flips up/down of the result."""
    if x_mirror and y_mirror:
        return cv2.flip(frame, -1)
    if x_mirror:
        return cv2.flip(frame, 1)
    if y_mirror:
        return cv2.flip(frame, 0)
    return frame


class HandTrackerService:
    def __init__(
        self,
        side_cam_index: int | None,
        top_cam_index: int | None,
        side_rotation_deg: int = 0,
        top_rotation_deg: int = 0,
        side_mirror: tuple[bool, bool] = (False, False),
        top_mirror: tuple[bool, bool] = (False, False),
    ):
        self.side_cam_index = side_cam_index
        self.top_cam_index = top_cam_index
        self._camera_rotation = {
            "side": int(side_rotation_deg) % 360,
            "top": int(top_rotation_deg) % 360,
        }
        self._camera_mirror = {"side": tuple(side_mirror), "top": tuple(top_mirror)}
        self._timing_sums = {"read": 0.0, "process": 0.0, "rerun": 0.0, "loop": 0.0}
        self._timing_count = 0
        self._timing_last_report = time.monotonic()
        self._rerun_image_tick = 0
        self._logger = _setup_run_logger()
        self._lock = threading.Lock()
        self._latest_gripper_fraction = 0.0
        self._latest_frames: dict[str, np.ndarray] = {"side": _black_frame(), "top": _black_frame()}
        self._latest_camera_fractions: dict[str, float | None] = {"side": None, "top": None}
        self._latest_xy: dict[str, dict[str, float] | None] = {"side": None, "top": None}
        self._running = False
        self._thread: threading.Thread | None = None
        self._caps: dict[str, cv2.VideoCapture | None] = {"side": None, "top": None}
        self._landmarkers: dict[str, hand_landmarker.HandLandmarker | None] = {"side": None, "top": None}
        self._calib = {
            "side": {"min_dist": 0.02, "max_dist": 0.18, "current": None, "smoothed": None},
            "top": {"min_dist": 0.02, "max_dist": 0.18, "current": None, "smoothed": None},
        }
        self._smoothing_alpha = 0.3
        self._start_times = {"side": None, "top": None}
        self._read_failure_limit = 10
        self._read_failure_counts = {"side": 0, "top": 0}
        self._top_xy_mapper = RelativeXYMapper()
        self._side_z_mapper = RelativeXYMapper()
        self._z_scale = DEFAULT_Z_SCALE_M
        self._xy_scale = DEFAULT_XY_SCALE_M

    def start(self) -> None:
        if self._running:
            return

        self._open_resources()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="HandTrackerService", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release_resources()

    def get_gripper_fraction(self) -> float:
        with self._lock:
            return float(self._latest_gripper_fraction)

    def get_latest_frames(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {label: frame.copy() for label, frame in self._latest_frames.items()}

    def get_latest_xy(self) -> dict[str, dict[str, float] | None]:
        with self._lock:
            return {label: (state.copy() if state is not None else None) for label, state in self._latest_xy.items()}

    def calibrate_open(self) -> None:
        with self._lock:
            for label in ("side", "top"):
                current = self._calib[label]["current"]
                if current is not None:
                    self._calib[label]["max_dist"] = current

    def calibrate_closed(self) -> None:
        with self._lock:
            for label in ("side", "top"):
                current = self._calib[label]["current"]
                if current is not None:
                    self._calib[label]["min_dist"] = current

    def calibrate_top_xy(self, midpoint: tuple[float, float] | None) -> None:
        if midpoint is None:
            return
        with self._lock:
            self._top_xy_mapper.reset(midpoint)

    def adjust_smoothing(self, delta: float) -> None:
        with self._lock:
            self._smoothing_alpha = min(1.0, max(0.05, self._smoothing_alpha + delta))

    def calibrate_side_z(self, midpoint: tuple[float, float] | None) -> None:
        if midpoint is None:
            return
        with self._lock:
            self._side_z_mapper.reset(midpoint)

    def adjust_z_scale(self, delta_m: float) -> float:
        with self._lock:
            self._z_scale = min(MAX_Z_SCALE_M, max(MIN_Z_SCALE_M, self._z_scale + delta_m))
            return float(self._z_scale)

    def get_z_scale(self) -> float:
        with self._lock:
            return float(self._z_scale)

    def _compute_relative_side_z(self, midpoint: tuple[float, float], frame_height: int) -> dict[str, float]:
        with self._lock:
            self._side_z_mapper.max_delta_px = MAX_DELTA_HEIGHT_FRACTION * frame_height
            self._side_z_mapper.deadzone_px = DEADZONE_HEIGHT_FRACTION * frame_height
            state = self._side_z_mapper.update(midpoint)
            scale = float(self._z_scale)
        # Image y points down, so a hand moving up (smaller y) is +z.
        return {
            "dz_px": float(-state["dy_px"]),
            "dz_norm": float(-state["dy_norm"]),
            "mid_x": float(state["mid_x"]),
            "mid_y": float(state["mid_y"]),
            "z_scale": scale,
        }

    def adjust_xy_scale(self, delta_m: float) -> float:
        with self._lock:
            self._xy_scale = min(MAX_XY_SCALE_M, max(MIN_XY_SCALE_M, self._xy_scale + delta_m))
            return float(self._xy_scale)

    def get_xy_scale(self) -> float:
        with self._lock:
            return float(self._xy_scale)

    def _open_resources(self) -> None:
        for label, cam_index in (("side", self.side_cam_index), ("top", self.top_cam_index)):
            self._read_failure_counts[label] = 0
            if cam_index is None:
                self._caps[label] = None
                self._landmarkers[label] = None
                continue

            cap = cv2.VideoCapture(cam_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._logger.info(
                    "%s camera %s opened at %dx%d",
                    label,
                    cam_index,
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            if not cap.isOpened():
                self._logger.warning("%s camera index %s could not be opened; using black frames.", label, cam_index)
                self._caps[label] = None
                self._landmarkers[label] = None
                continue

            self._caps[label] = cap
            self._landmarkers[label] = _make_landmarker()

    def _release_resources(self) -> None:
        for label in ("side", "top"):
            cap = self._caps[label]
            if cap is not None:
                cap.release()
            landmarker = self._landmarkers[label]
            if landmarker is not None:
                landmarker.close()
            self._caps[label] = None
            self._landmarkers[label] = None

    def _get_timestamp_ms(self, label: str) -> int:
        now = time.monotonic()
        if self._start_times[label] is None:
            self._start_times[label] = now
        return int((now - float(self._start_times[label])) * 1000)

    def _apply_smoothing(self, label: str, raw_value: float) -> float:
        prev = self._calib[label]["smoothed"]
        if prev is None:
            smoothed = raw_value
        else:
            smoothed = self._smoothing_alpha * raw_value + (1 - self._smoothing_alpha) * float(prev)
        self._calib[label]["smoothed"] = smoothed
        return smoothed

    def _compute_midpoint(self, points: dict[int, tuple[int, int]], idx1: int, idx2: int) -> tuple[float, float] | None:
        if idx1 in points and idx2 in points:
            x1, y1 = points[idx1]
            x2, y2 = points[idx2]
            return (x1 + x2) / 2, (y1 + y2) / 2
        return None

    def _snap_orientation_to_180(self, orientation_vector: tuple[float, float] | None) -> tuple[tuple[float, float], float] | None:
        if orientation_vector is None:
            return None

        vx, vy = orientation_vector
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            return None

        # Convert from image coordinates (y down) into a user/cartesian frame
        # where +Y is upward and 0 degrees means pointing upward.
        user_x = vx / norm
        user_y = -vy / norm
        angle_deg = (math.degrees(math.atan2(user_x, user_y)) + 360.0) % 360.0
        snapped_deg = (round(angle_deg / 180.0) * 180.0) % 360.0
        snapped_rad = math.radians(snapped_deg)

        snapped_user_x = math.sin(snapped_rad)
        snapped_user_y = math.cos(snapped_rad)

        # Return back in image coordinates for display (y down).
        snapped_img_x = float(snapped_user_x)
        snapped_img_y = float(-snapped_user_y)
        return (snapped_img_x, snapped_img_y), snapped_deg

    def _compute_relative_top_xy(
        self,
        midpoint: tuple[float, float],
        orientation_vector: tuple[float, float] | None,
        frame_width: int,
    ) -> dict[str, float]:
        with self._lock:
            self._top_xy_mapper.max_delta_px = MAX_DELTA_FRAME_FRACTION * frame_width
            self._top_xy_mapper.deadzone_px = DEADZONE_FRAME_FRACTION * frame_width
            state = self._top_xy_mapper.update(midpoint)
            scale = float(self._xy_scale)

        snapped_orientation = self._snap_orientation_to_180(orientation_vector)

        # Auto-orient axes so +Y follows wrist->finger direction.
        # Fallback to camera axes when orientation is unavailable.
        if snapped_orientation is None:
            ux_x, ux_y = 1.0, 0.0
            uy_x, uy_y = 0.0, 1.0
            snapped_deg = None
        else:
            _, snapped_deg = snapped_orientation
            snapped_rad = math.radians(snapped_deg)
            # User frame: +Y is the hand's wrist->finger direction, +X is the
            # clockwise perpendicular so left/right remain intuitive on the
            # overhead view.
            uy_x, uy_y = math.sin(snapped_rad), math.cos(snapped_rad)
            ux_x, ux_y = math.cos(snapped_rad), -math.sin(snapped_rad)

        raw_dx = float(state.get("dx_px", 0.0))
        raw_dy = float(state.get("dy_px", 0.0))

        # Convert raw image-space deltas into the user/cartesian frame.
        raw_user_dx = raw_dx
        raw_user_dy = -raw_dy

        aligned_x_px = raw_user_dx * ux_x + raw_user_dy * ux_y
        aligned_y_px = raw_user_dx * uy_x + raw_user_dy * uy_y

        max_delta = float(self._top_xy_mapper.max_delta_px)
        deadzone = float(self._top_xy_mapper.deadzone_px)

        if abs(aligned_x_px) < deadzone:
            aligned_x_px = 0.0
        if abs(aligned_y_px) < deadzone:
            aligned_y_px = 0.0

        aligned_x_px = max(-max_delta, min(max_delta, aligned_x_px))
        aligned_y_px = max(-max_delta, min(max_delta, aligned_y_px))

        dx_norm = aligned_x_px / max_delta if max_delta > 0 else 0.0
        dy_norm = aligned_y_px / max_delta if max_delta > 0 else 0.0

        dx_norm = max(-1.0, min(1.0, dx_norm))
        dy_norm = max(-1.0, min(1.0, dy_norm))

        state["dx_px"] = float(aligned_x_px)
        state["dy_px"] = float(aligned_y_px)
        state["dx_norm"] = float(dx_norm)
        state["dy_norm"] = float(dy_norm)
        state["xy_scale"] = float(scale)
        if snapped_deg is not None:
            state["orientation_deg"] = float(snapped_deg)
        return state

    def _process_frame(self, frame: np.ndarray, landmarker: hand_landmarker.HandLandmarker, label: str) -> tuple[np.ndarray, float | None]:
        display_frame = frame.copy()
        display_frame = _rotate_frame(display_frame, self._camera_rotation.get(label, 0))
        x_mirror, y_mirror = self._camera_mirror.get(label, (False, False))
        display_frame = _mirror_frame(display_frame, x_mirror, y_mirror)
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = self._get_timestamp_ms(label)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        connections = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS
        height, width = display_frame.shape[:2]

        norm_val_to_show = None
        dist_cm_to_show = None
        top_midpoint = None
        side_midpoint = None
        top_orientation_vector = None
        top_snapped_orientation = None

        def _draw_top_xy_axes() -> None:
            if label != "top":
                return

            with self._lock:
                origin = self._top_xy_mapper.reference

            if origin is None:
                origin = (width * 0.5, height * 0.5)

            origin_x, origin_y = int(origin[0]), int(origin[1])
            axis_len = max(40, min(width, height) // 5)

            if top_snapped_orientation is None:
                y_axis = (0.0, -1.0)
            else:
                y_axis = top_snapped_orientation[0]

            x_axis = (float(y_axis[1]), float(-y_axis[0]))
            x_end = (int(origin_x + x_axis[0] * axis_len), int(origin_y + x_axis[1] * axis_len))
            y_end = (int(origin_x + y_axis[0] * axis_len), int(origin_y + y_axis[1] * axis_len))

            cv2.circle(display_frame, (origin_x, origin_y), 7, (0, 255, 255), -1)
            cv2.putText(display_frame, "origin", (origin_x + 8, origin_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.arrowedLine(display_frame, (origin_x, origin_y), x_end, (255, 120, 120), 2, tipLength=0.18)
            cv2.arrowedLine(display_frame, (origin_x, origin_y), y_end, (120, 255, 120), 2, tipLength=0.18)
            cv2.putText(display_frame, "X", (x_end[0] + 5, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 120), 2)
            cv2.putText(display_frame, "Y", (y_end[0] + 5, y_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)

        for hand_landmarks, hand_world_landmarks in zip(results.hand_landmarks, results.hand_world_landmarks):
            points: dict[int, tuple[int, int]] = {}
            for idx, landmark in enumerate(hand_landmarks):
                if idx in ALLOWED_LANDMARKS:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points[idx] = (x, y)
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)

            for connection in connections:
                if connection.start in ALLOWED_LANDMARKS and connection.end in ALLOWED_LANDMARKS:
                    if connection.start in points and connection.end in points:
                        cv2.line(display_frame, points[connection.start], points[connection.end], (255, 0, 0), 2)

            if label == "side":
                side_midpoint = self._compute_midpoint(points, 2, 5)
                if side_midpoint is not None:
                    cv2.circle(display_frame, (int(side_midpoint[0]), int(side_midpoint[1])), 6, (255, 255, 0), -1)

            if label == "top":
                top_midpoint = self._compute_midpoint(points, 2, 5)
                if top_midpoint is not None:
                    cv2.circle(display_frame, (int(top_midpoint[0]), int(top_midpoint[1])), 6, (255, 255, 0), -1)

                wrist = points.get(0)
                finger_root = points.get(5)
                if wrist is not None and finger_root is not None:
                    top_orientation_vector = (float(finger_root[0] - wrist[0]), float(finger_root[1] - wrist[1]))
                    top_snapped_orientation = self._snap_orientation_to_180(top_orientation_vector)
                    arrow_vector = top_orientation_vector
                    if top_snapped_orientation is not None:
                        arrow_vector = top_snapped_orientation[0]

                    arrow_len = max(30, min(width, height) // 6)
                    arrow_end = (
                        int(wrist[0] + arrow_vector[0] * arrow_len),
                        int(wrist[1] + arrow_vector[1] * arrow_len),
                    )
                    cv2.arrowedLine(display_frame, (int(wrist[0]), int(wrist[1])), arrow_end, (0, 200, 255), 2, tipLength=0.2)

                _draw_top_xy_axes()

            if 4 in points and 8 in points:
                p_thumb, p_index = points[4], points[8]
                w_thumb = hand_world_landmarks[4]
                w_index = hand_world_landmarks[8]
                raw_dist = math.sqrt(
                    (w_thumb.x - w_index.x) ** 2
                    + (w_thumb.y - w_index.y) ** 2
                    + (w_thumb.z - w_index.z) ** 2
                )

                smoothed_dist = self._apply_smoothing(label, raw_dist)
                self._calib[label]["current"] = smoothed_dist

                min_d, max_d = self._calib[label]["min_dist"], self._calib[label]["max_dist"]
                norm_val = 0.0
                if max_d > min_d:
                    norm_val = max(0.0, min(1.0, (smoothed_dist - min_d) / (max_d - min_d)))

                cv2.line(display_frame, p_thumb, p_index, (0, 255, 255), 3)
                norm_val_to_show = norm_val
                dist_cm_to_show = smoothed_dist * 100

        if norm_val_to_show is not None:
            status_text = f"{label.upper()}  Range: {norm_val_to_show:.2f}   Dist: {dist_cm_to_show:.1f} cm"
            color = (0, 255, 255)
        else:
            status_text = f"{label.upper()}  (no hand detected)"
            color = (0, 0, 255)

        if label == "top":
            rel_xy = None
            if top_midpoint is not None:
                rel_xy = self._compute_relative_top_xy(top_midpoint, top_orientation_vector, width)
            with self._lock:
                self._latest_xy["top"] = rel_xy
                scale_now = float(self._xy_scale)

            dx_norm_val = float(rel_xy.get("dx_norm", 0.0)) if rel_xy is not None else 0.0
            dy_norm_val = float(rel_xy.get("dy_norm", 0.0)) if rel_xy is not None else 0.0

            if top_snapped_orientation is not None and top_snapped_orientation[1] is not None:
                status_text += f"   Dir: {top_snapped_orientation[1]:.0f}°"

            scale_text = f"XY scale {scale_now:.3f} m"
            norms_text = f"dx {dx_norm_val:.2f}  dy {dy_norm_val:.2f}"
            cv2.rectangle(display_frame, (15, 108), (380, 225), (0, 0, 0), -1)
            cv2.putText(display_frame, norms_text, (22, 148), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            cv2.putText(display_frame, scale_text, (22, 182), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
            if top_snapped_orientation is not None and top_snapped_orientation[1] is not None:
                dir_text = f"DIR {top_snapped_orientation[1]:.0f}°"
                cv2.putText(display_frame, dir_text, (22, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 180, 80), 2)

        if label == "side":
            rel_z = None
            if side_midpoint is not None:
                rel_z = self._compute_relative_side_z(side_midpoint, height)
            with self._lock:
                self._latest_xy["side"] = rel_z
                z_scale_now = float(self._z_scale)
                z_ref = self._side_z_mapper.reference

            dz_norm_val = float(rel_z.get("dz_norm", 0.0)) if rel_z is not None else 0.0
            if z_ref is not None:
                ref_y = int(z_ref[1])
                cv2.line(display_frame, (0, ref_y), (width - 1, ref_y), (0, 255, 255), 1)
                cv2.putText(display_frame, "z ref", (width - 90, max(ref_y - 6, 120)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.rectangle(display_frame, (15, 108), (380, 190), (0, 0, 0), -1)
            cv2.putText(display_frame, f"dz {dz_norm_val:+.2f}", (22, 148), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            cv2.putText(display_frame, f"Z scale {z_scale_now:.3f} m", (22, 182), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)

        cv2.rectangle(display_frame, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.putText(display_frame, status_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 2)

        return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), norm_val_to_show

    def _safe_process_frame(
        self,
        frame: np.ndarray,
        landmarker: hand_landmarker.HandLandmarker,
        label: str,
    ) -> tuple[np.ndarray, float | None]:
        try:
            return self._process_frame(frame, landmarker, label)
        except Exception:
            self._logger.exception("%s frame processing failed", label)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None

    def _choose_gripper_fraction(self, side_fraction: float | None, top_fraction: float | None) -> float:
        if top_fraction is not None: #prioritize top camera for now
            return top_fraction
        if side_fraction is not None:
            return side_fraction
        return self._latest_gripper_fraction

    def _record_timing(self, read_s: float, process_s: float, rerun_s: float, loop_s: float) -> None:
        sums = self._timing_sums
        sums["read"] += read_s
        sums["process"] += process_s
        sums["rerun"] += rerun_s
        sums["loop"] += loop_s
        self._timing_count += 1
        now = time.monotonic()
        if now - self._timing_last_report < 5.0:
            return
        n = max(1, self._timing_count)
        self._logger.info(
            "Handtracker loop avg over %d iters: read=%.1fms process(mediapipe+draw)=%.1fms rerun=%.1fms loop=%.1fms (%.1f Hz)",
            n,
            sums["read"] / n * 1000,
            sums["process"] / n * 1000,
            sums["rerun"] / n * 1000,
            sums["loop"] / n * 1000,
            n / (now - self._timing_last_report),
        )
        top_state = self.get_latest_xy().get("top")
        with self._lock:
            top_shape = self._latest_frames["top"].shape
        reference = self._top_xy_mapper.reference
        if isinstance(top_state, dict):
            self._logger.info(
                "Top XY: frame=%dx%d max_delta_px=%.0f ref=%s mid=(%.0f,%.0f) dx_px=%.1f dy_px=%.1f dx_norm=%+.2f dy_norm=%+.2f",
                top_shape[1],
                top_shape[0],
                self._top_xy_mapper.max_delta_px,
                None if reference is None else f"({reference[0]:.0f},{reference[1]:.0f})",
                top_state.get("mid_x", 0.0),
                top_state.get("mid_y", 0.0),
                top_state.get("dx_px", 0.0),
                top_state.get("dy_px", 0.0),
                top_state.get("dx_norm", 0.0),
                top_state.get("dy_norm", 0.0),
            )
        else:
            self._logger.info("Top XY: frame=%dx%d hand NOT detected", top_shape[1], top_shape[0])
        side_state = self.get_latest_xy().get("side")
        if isinstance(side_state, dict):
            self._logger.info(
                "Side Z: dz_px=%.1f dz_norm=%+.2f z_scale=%.3f", side_state.get("dz_px", 0.0), side_state.get("dz_norm", 0.0), side_state.get("z_scale", 0.0)
            )
        else:
            self._logger.info("Side Z: hand NOT detected")
        for key in sums:
            sums[key] = 0.0
        self._timing_count = 0
        self._timing_last_report = now

    def _log_rerun(self, side_frame: np.ndarray, top_frame: np.ndarray) -> None:
        try:
            import rerun as rr
        except Exception as e:
            self._logger.warning("Failed to import rerun: %s", e)
            return

        top_state = self.get_latest_xy().get("top")
        detected = isinstance(top_state, dict)
        dx_norm = float(top_state.get("dx_norm", 0.0)) if detected else 0.0
        dy_norm = float(top_state.get("dy_norm", 0.0)) if detected else 0.0
        scale = self.get_xy_scale()
        side_state = self.get_latest_xy().get("side")
        side_detected = isinstance(side_state, dict)
        dz_norm = float(side_state.get("dz_norm", 0.0)) if side_detected else 0.0
        z_scale = self.get_z_scale()

        rr.log("hand_tracker/dz_norm", rr.Scalars(dz_norm))
        rr.log("hand_tracker/z_scale", rr.Scalars(z_scale))
        rr.log("hand_tracker/dx_norm", rr.Scalars(dx_norm))
        rr.log("hand_tracker/dy_norm", rr.Scalars(dy_norm))
        rr.log("hand_tracker/xy_scale", rr.Scalars(scale))
        rr.log("hand_tracker/hand_detected", rr.Scalars(1.0 if detected else 0.0))
        rr.log(
            "hand_tracker/status",
            rr.TextDocument(
                f"hand {'detected' if detected else 'NOT detected'}\n\n"
                f"dx_norm = {dx_norm:+.3f}\n\ndy_norm = {dy_norm:+.3f}\n\nxy_scale = {scale:.3f} m\n\n"
                f"side hand {'detected' if side_detected else 'NOT detected'}: dz_norm = {dz_norm:+.3f}\n\nz_scale = {z_scale:.3f} m"
            ),
        )

        self._rerun_image_tick += 1
        if _RERUN_LOGGING_MODE == "full" and self._rerun_image_tick % RERUN_IMAGE_EVERY_N == 0:
            try:
                rr.log("hand_tracker/side", rr.Image(side_frame).compress(jpeg_quality=70))
                rr.log("hand_tracker/top", rr.Image(top_frame).compress(jpeg_quality=70))
                rr.log("hand_tracker/gripper_fraction", rr.Scalars(self.get_gripper_fraction()))
            except Exception as e:
                self._logger.warning("Rerun image/gripper log failed: %s", e)

    def _run_loop(self) -> None:
        target_dt = 1.0 / 15.0
        while self._running:
            loop_start = time.monotonic()
            side_frame = _black_frame()
            top_frame = _black_frame()
            side_fraction = None
            top_fraction = None
            read_time = 0.0

            side_cap = self._caps["side"]
            side_landmarker = self._landmarkers["side"]
            if side_cap is not None and side_landmarker is not None:
                t_read = time.monotonic()
                ok, frame = side_cap.read()
                read_time += time.monotonic() - t_read
                if ok and _is_valid_frame(frame):
                    self._read_failure_counts["side"] = 0
                    side_frame, side_fraction = self._safe_process_frame(frame, side_landmarker, "side")
                else:
                    self._read_failure_counts["side"] += 1
                    self._logger.warning(
                        "Side camera frame read failed (%s/%s); retrying...",
                        self._read_failure_counts["side"],
                        self._read_failure_limit,
                    )
                    if self._read_failure_counts["side"] >= self._read_failure_limit:
                        self._logger.warning("Side camera exceeded retry limit; disabling side camera.")
                        side_cap.release()
                        self._caps["side"] = None
                        self._landmarkers["side"] = None
                    side_frame = self._latest_frames.get("side", _black_frame())

            top_cap = self._caps["top"]
            top_landmarker = self._landmarkers["top"]
            if top_cap is not None and top_landmarker is not None:
                t_read = time.monotonic()
                ok, frame = top_cap.read()
                read_time += time.monotonic() - t_read
                if ok and _is_valid_frame(frame):
                    self._read_failure_counts["top"] = 0
                    top_frame, top_fraction = self._safe_process_frame(frame, top_landmarker, "top")
                else:
                    self._read_failure_counts["top"] += 1
                    self._logger.warning(
                        "Top camera frame read failed (%s/%s); retrying...",
                        self._read_failure_counts["top"],
                        self._read_failure_limit,
                    )
                    if self._read_failure_counts["top"] >= self._read_failure_limit:
                        self._logger.warning("Top camera exceeded retry limit; disabling top camera.")
                        top_cap.release()
                        self._caps["top"] = None
                        self._landmarkers["top"] = None
                    top_frame = self._latest_frames.get("top", _black_frame())

            with self._lock:
                self._latest_frames["side"] = side_frame
                self._latest_frames["top"] = top_frame
                self._latest_camera_fractions["side"] = side_fraction
                self._latest_camera_fractions["top"] = top_fraction
                self._latest_gripper_fraction = self._choose_gripper_fraction(side_fraction, top_fraction)

            t_process_end = time.monotonic()
            t_rerun_start = t_process_end
            if _RERUN_LOGGING_ENABLED and _RERUN_LOGGING_MODE in {"lightweight", "full"}:
                self._log_rerun(side_frame, top_frame)
            t_rerun_end = time.monotonic()

            elapsed = time.monotonic() - loop_start
            self._record_timing(read_time, (t_process_end - loop_start) - read_time, t_rerun_end - t_rerun_start, elapsed)
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            if self._caps["side"] is None and self._caps["top"] is None:
                time.sleep(0.05)


_default_service: HandTrackerService | None = None


def init_default_service(
    side_cam_index: int | None,
    top_cam_index: int | None,
    side_rotation_deg: int = 0,
    top_rotation_deg: int = 0,
    side_mirror: tuple[bool, bool] = (False, False),
    top_mirror: tuple[bool, bool] = (False, False),
) -> None:
    global _default_service
    if _default_service is not None:
        _default_service.stop()
    _default_service = HandTrackerService(
        side_cam_index, top_cam_index, side_rotation_deg, top_rotation_deg, side_mirror, top_mirror
    )
    _default_service.start()


def get_gripper_fraction() -> float:
    if _default_service is None:
        return 0.0
    return _default_service.get_gripper_fraction()


def get_latest_frames() -> dict[str, np.ndarray]:
    if _default_service is None:
        return {"side": _black_frame(), "top": _black_frame()}
    return _default_service.get_latest_frames()


def get_latest_xy() -> dict[str, dict[str, float] | None]:
    if _default_service is None:
        return {"side": None, "top": None}
    return _default_service.get_latest_xy()


def calibrate_open() -> None:
    if _default_service is not None:
        _default_service.calibrate_open()


def calibrate_closed() -> None:
    if _default_service is not None:
        _default_service.calibrate_closed()


def calibrate_top_xy(midpoint: tuple[float, float] | None) -> None:
    if _default_service is not None and midpoint is not None:
        _default_service.calibrate_top_xy(midpoint)


def adjust_smoothing(delta: float) -> None:
    if _default_service is not None:
        _default_service.adjust_smoothing(delta)


def calibrate_side_z(midpoint: tuple[float, float] | None) -> None:
    if _default_service is not None and midpoint is not None:
        _default_service.calibrate_side_z(midpoint)


def adjust_z_scale(delta_m: float) -> float:
    if _default_service is not None:
        return _default_service.adjust_z_scale(delta_m)
    return DEFAULT_Z_SCALE_M


def get_z_scale() -> float:
    if _default_service is not None:
        return _default_service.get_z_scale()
    return DEFAULT_Z_SCALE_M


def adjust_xy_scale(delta_m: float) -> float:
    if _default_service is not None:
        return _default_service.adjust_xy_scale(delta_m)
    return DEFAULT_XY_SCALE_M


def get_xy_scale() -> float:
    if _default_service is not None:
        return _default_service.get_xy_scale()
    return DEFAULT_XY_SCALE_M


def shutdown_default_service() -> None:
    global _default_service
    if _default_service is not None:
        _default_service.stop()
        _default_service = None


def _parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null", ""}:
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Background MediaPipe hand tracker service")
    parser.add_argument("--side-cam", type=_parse_optional_int, default=0)
    parser.add_argument("--top-cam", type=_parse_optional_int, default=1)
    args = parser.parse_args()

    _setup_run_logger()
    init_default_service(args.side_cam, args.top_cam)
    _RUN_LOGGER.info("HandTrackerService started. Press Ctrl-C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_default_service()


if __name__ == "__main__":
    main()