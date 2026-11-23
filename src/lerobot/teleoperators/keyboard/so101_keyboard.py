#!/usr/bin/env python

"""Keyboard teleoperator that produces SO101 end-effector targets.

Two modes are supported:
- Keyboard-only: arrows control `ee.x` / `ee.y`, `w` / `s` control `ee.z`.
- Handtracker + keyboard: the top camera controls `ee.x` / `ee.y`, the side camera
    controls `ee.z` (hand height). `w` / `s` shift the z reference. `q` / `e` tune the
    XY scale and `,` / `.` tune the Z scale.

In both modes, `1` = keyboard mode, `2` = lightweight handtracker logging,
`3` = full handtracker video logging, and `q` / `e` tune the active handtracker
motion scale while in handtracker mode.

In handtracker modes (`2` / `3`), `4` toggles the XY mapping, `5` toggles the Z
mapping, and `6` cycles pitch mapping: 2D elevation -> 3D elevation -> off.
A disabled axis group keeps its last position. Re-enabling re-bases the hand
reference to the current hand pose, so the arm does not jump.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.utils.import_utils import require_package
from lerobot.utils.rotation import Rotation
from ..teleoperator import Teleoperator
from ..config import TeleoperatorConfig

logger = logging.getLogger(__name__)

EE_POSE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")
EE_GRIPPER_KEY = "ee.gripper_pos"
CONTROL_MODE_KEYBOARD = "keyboard"
CONTROL_MODE_HANDTRACKER = "handtracker"
XY_SCALE_STEP_M = 0.01
MAX_XY_SCALE_M = 0.40
# User-frame XY limits (+y = forward in the overhead view). The sim accepts a
# +/-0.55 m box with a max TCP reach of ~0.5457 m. Measured on the sim: sideways
# +/-0.40 m and forward up to ~0.45 m are reached; the area close to the base is not.
EE_X_LIMIT_M = 0.40
EE_Y_MIN_M = -0.25
EE_Y_MAX_M = 0.45
EE_Z_MIN_M = 0.02
EE_Z_MAX_M = 0.40
Z_SCALE_STEP_M = 0.01
LOCKED_ORIENTATION = {"ee.wx": 0.0, "ee.wy": 0.0, "ee.wz": 0.0}
PITCH_MODE_OFF = "off"
PITCH_MODE_2D = "2d"
PITCH_MODE_3D = "3d"
PITCH_DEADZONE_DEG = 2.0
PITCH_CLAMP_DEG = 40.0
PITCH_EMA_TAU_S = 0.15
PITCH_RATE_LIMIT_DEG_PER_S = 120.0
PITCH_GAIN_DEFAULT = 1.0
PITCH_DOWN_MIN_DEG = 0.0
PITCH_DOWN_MAX_DEG = 90.0
PITCH_DOWN_DEFAULT_DEG = 45.0
# Pitch tick timing: the dt used by the EMA / rate limit is refreshed every tick and
# clamped, so the first update after a gap cannot bypass the rate limit.
PITCH_DT_MIN_S = 1e-3
PITCH_DT_MAX_S = 0.1
# If the pitch measurement has been unavailable for longer than this, re-anchor the
# reference to the hand's current elevation when it comes back.
PITCH_REBASE_TIMEOUT_S = 1.0


@TeleoperatorConfig.register_subclass("keyboard_so_leader")
@dataclass
class KeyboardSOLeaderConfig(TeleoperatorConfig):
    """Config for the keyboard SO leader teleoperator."""

    step_deg: float = 1.0
    use_degrees: bool = True
    handtracker_module: str | None = None
    # Extra roll applied to the seeded start orientation, about the tool (gripper) axis.
    # The sim follower starts with its wrist 90 deg anticlockwise from upright; +90
    # (clockwise, looking along the gripper) puts the wrist camera upright again.
    wrist_roll_offset_deg: float = 90.0


class KeyboardSOLeader(Teleoperator):
    """Keyboard teleop that returns end-effector targets for SO101.

    Designed for interactive testing in simulator where a hardware leader is
    not available. The action keys are intentionally pose-centric.
    """

    config_class = KeyboardSOLeaderConfig
    name = "keyboard_so_leader"

    def __init__(self, config: KeyboardSOLeaderConfig):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.motors = [*EE_POSE_KEYS, EE_GRIPPER_KEY]
        # Internal absolute state in end-effector units (metres/radians for pose,
        # [0,100] normalized for gripper).
        self._state: dict[str, float] = {key: 0.0 for key in self.motors}
        self._state["ee.z"] = 0.15
        self._listener = None
        self._control_mode: str = CONTROL_MODE_KEYBOARD
        # 1 = keyboard, 2 = handtracker (lightweight logging), 3 = handtracker (full logging)
        self._active_mode: int = 1
        self._last_key: str | None = None
        self._motion_baseline: dict[str, float] = {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.15}
        self._xy_scale: float = 0.15
        self._keyboard_step: float = self.config.step_deg * 0.01
        self._last_relative_xy: dict[str, float] = {"dx_px": 0.0, "dy_px": 0.0, "dx_norm": 0.0, "dy_norm": 0.0}
        self._last_xy_valid_t: float = 0.0
        self._last_relative_z: float = 0.0
        self._xy_mapping_enabled: bool = True
        self._z_mapping_enabled: bool = True
        self._xy_rebase_pending: bool = False
        self._z_rebase_pending: bool = False
        self._last_z_valid_t: float = 0.0
        self._z_scale: float = 0.15
        self._xy_hold_timeout_s: float = 0.35
        self._last_hand_log_t: float = 0.0
        self._hand_log_interval_s: float = 0.5
        self._orientation_lock = dict(LOCKED_ORIENTATION)
        self._apply_orientation_lock()
        self._pitch_mapping_mode: str = PITCH_MODE_OFF
        self._pitch_rebase_pending: bool = False
        self._pitch_gain: float = PITCH_GAIN_DEFAULT
        self._pitch_down_cmd_deg: float = PITCH_DOWN_DEFAULT_DEG
        self._pitch_down_baseline_deg: float = self._pitch_down_cmd_deg
        self._pitch_elevation_ref_deg: float | None = None
        self._last_pitch_delta_ema_deg: float = 0.0
        self._last_pitch_valid_t: float = 0.0
        self._last_pitch_measurement_deg: float = 0.0
        self._pitch_hold_timeout_s: float = self._xy_hold_timeout_s
        self._last_pitch_cmd_update_t: float = time.monotonic()
        self._pitch_gap_s: float = 0.0
        # Seeded start pose and the per-session pitch frame (delta-from-seed).
        self._seed_position: dict[str, float] = {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.15}
        self._seed_orientation_lock: dict[str, float] = dict(LOCKED_ORIENTATION)
        self._pitch_R_base: np.ndarray | None = None
        self._pitch_axis_n: np.ndarray | None = None
        self._pitch_down_seed_deg: float = PITCH_DOWN_DEFAULT_DEG

        # Optional handtracker integration
        self._handtracker = None
        if self.config.handtracker_module:
            try:
                import importlib

                self._handtracker = importlib.import_module(self.config.handtracker_module)
                logger.info("Loaded handtracker module %s", self.config.handtracker_module)
            except Exception as e:
                logger.warning("Could not import handtracker module %s: %s", self.config.handtracker_module, e)

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys((*EE_POSE_KEYS, EE_GRIPPER_KEY), float)

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        # No configuration required for the keyboard stub
        return None

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._listener is not None and getattr(self._listener, "is_alive", lambda: False)()

    @property
    def is_calibrated(self) -> bool:
        return True

    def _clamp_state(self, key: str, value: float) -> float:
        if key == "ee.z":
            return max(EE_Z_MIN_M, min(EE_Z_MAX_M, value))
        if key == EE_GRIPPER_KEY:
            return max(0.0, min(100.0, value))
        if key in self._orientation_lock:
            return self._orientation_lock[key]
        if key == "ee.y":
            return max(EE_Y_MIN_M, min(EE_Y_MAX_M, value))
        return max(-EE_X_LIMIT_M, min(EE_X_LIMIT_M, value))

    def _apply_orientation_lock(self) -> None:
        for key, value in self._orientation_lock.items():
            self._state[key] = float(value)

    def _set_handtracker_logging_mode(self, mode: str) -> None:
        # Only select the Rerun logging MODE here. The global "display enabled" flag is
        # owned by the --display-data path and must never be flipped from the keyboard,
        # otherwise pressing 2/3 under --no-display-data would try to log with no session.
        if self._handtracker is None:
            return
        setter = getattr(self._handtracker, "set_rerun_logging_mode", None)
        if callable(setter):
            setter(mode)

    def _set_control_mode(self, mode: str) -> None:
        if mode == self._control_mode:
            return
        if mode == CONTROL_MODE_HANDTRACKER:
            self._motion_baseline = {
                "ee.x": float(self._state["ee.x"]),
                "ee.y": float(self._state["ee.y"]),
                "ee.z": float(self._state["ee.z"]),
            }
            self._xy_rebase_pending = True
            self._z_rebase_pending = True
            self._pitch_mapping_mode = PITCH_MODE_2D
            self._pitch_rebase_pending = True
            self._pitch_down_baseline_deg = float(self._pitch_down_cmd_deg)
        self._control_mode = mode
        logger.info("Control mode switched to: %s", self._control_mode)

    @staticmethod
    def _rot_axis_angle(axis: np.ndarray, theta_rad: float) -> np.ndarray:
        """Rodrigues rotation matrix about a (not necessarily unit) axis."""
        k = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(k))
        if norm < 1e-12:
            return np.eye(3)
        k = k / norm
        skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return np.eye(3) + math.sin(theta_rad) * skew + (1.0 - math.cos(theta_rad)) * (skew @ skew)

    def _init_pitch_frame(self) -> None:
        """Derive the per-session pitch frame from the seeded (locked) orientation.

        The pitch command is then a rotation of the seed pose about a fixed horizontal
        axis ``n``, so at ``pitch_cmd == pitch_down_seed`` the target equals the seed
        orientation exactly (no jump when entering modes 2/3).
        """
        rotvec = np.array(
            [
                float(self._orientation_lock["ee.wx"]),
                float(self._orientation_lock["ee.wy"]),
                float(self._orientation_lock["ee.wz"]),
            ]
        )
        r_base = Rotation.from_rotvec(rotvec).as_matrix()
        approach = r_base[:, 2]  # tool z axis in world = gripper approach axis
        self._pitch_R_base = r_base
        self._pitch_down_seed_deg = math.degrees(math.asin(max(-1.0, min(1.0, -float(approach[2])))))
        azimuth = math.atan2(float(approach[1]), float(approach[0]))
        self._pitch_axis_n = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
        self._pitch_down_cmd_deg = float(self._pitch_down_seed_deg)
        self._pitch_down_baseline_deg = float(self._pitch_down_seed_deg)
        self._last_pitch_delta_ema_deg = 0.0
        # Self-check: at the seed pitch the target must reproduce the seed orientation.
        check = self._rot_axis_angle(self._pitch_axis_n, 0.0) @ r_base
        err = Rotation.from_matrix(np.asarray(check, dtype=float)).inv() * Rotation.from_matrix(
            np.asarray(r_base, dtype=float)
        )
        logger.info(
            "Pitch frame: seed pitch_down=%.2f deg, azimuth=%.1f deg, axis=(%.3f, %.3f, %.3f), "
            "approach=(%.3f, %.3f, %.3f), self-check=%.4f deg",
            self._pitch_down_seed_deg,
            math.degrees(azimuth),
            float(self._pitch_axis_n[0]),
            float(self._pitch_axis_n[1]),
            float(self._pitch_axis_n[2]),
            float(approach[0]),
            float(approach[1]),
            float(approach[2]),
            math.degrees(float(np.linalg.norm(err.as_rotvec()))),
        )

    def seed_initial_pose(self, pose: dict[str, float]) -> None:
        aliases = {
            "ee.x": ("ee.x", "x"),
            "ee.y": ("ee.y", "y"),
            "ee.z": ("ee.z", "z"),
            "ee.wx": ("ee.wx", "wx"),
            "ee.wy": ("ee.wy", "wy"),
            "ee.wz": ("ee.wz", "wz"),
            EE_GRIPPER_KEY: (EE_GRIPPER_KEY, "gripper", "gripper_pos"),
        }
        orientation_keys = {"ee.wx", "ee.wy", "ee.wz"}
        for target_key, source_keys in aliases.items():
            for source_key in source_keys:
                if source_key in pose:
                    raw_value = float(pose[source_key])
                    if target_key in orientation_keys:
                        # Preserve measured orientation while seeding; we refresh
                        # the orientation lock from this seeded value below.
                        self._state[target_key] = raw_value
                    else:
                        self._state[target_key] = self._clamp_state(target_key, raw_value)
                    break
        self._motion_baseline = {
            "ee.x": float(self._state["ee.x"]),
            "ee.y": float(self._state["ee.y"]),
            "ee.z": float(self._state["ee.z"]),
        }
        measured = (float(self._state["ee.wx"]), float(self._state["ee.wy"]), float(self._state["ee.wz"]))
        self._orientation_lock = self._rolled_orientation(measured, self.config.wrist_roll_offset_deg)
        self._seed_orientation_lock = dict(self._orientation_lock)
        self._seed_position = {
            "ee.x": float(self._state["ee.x"]),
            "ee.y": float(self._state["ee.y"]),
            "ee.z": float(self._state["ee.z"]),
        }
        self._init_pitch_frame()
        self._apply_orientation_lock()
        logger.info("Seeded initial EE pose: %s", {k: self._state[k] for k in self.motors})

    @staticmethod
    def _rolled_orientation(rotvec: tuple[float, float, float], roll_deg: float) -> dict[str, float]:
        """Rotate a world-frame orientation (rotation vector) about its own tool z axis."""
        if roll_deg == 0.0:
            wx, wy, wz = rotvec
        else:
            rolled = Rotation.from_rotvec(np.asarray(rotvec)) * Rotation.from_rotvec(
                np.array([0.0, 0.0, math.radians(roll_deg)])
            )
            wx, wy, wz = (float(v) for v in rolled.as_rotvec())
        return {"ee.wx": wx, "ee.wy": wy, "ee.wz": wz}

    def _reset_pose(self) -> None:
        self._state = {key: 0.0 for key in self.motors}
        self._state["ee.z"] = 0.15
        self._apply_orientation_lock()
        self._motion_baseline = {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.15}
        logger.info("Reset end-effector targets to defaults")

    def _move_keyboard_xy(self, dx: float, dy: float) -> None:
        self._state["ee.x"] = self._clamp_state("ee.x", self._state["ee.x"] + dx)
        self._state["ee.y"] = self._clamp_state("ee.y", self._state["ee.y"] + dy)
        logger.info("Keyboard XY -> ee.x=%.3f ee.y=%.3f", self._state["ee.x"], self._state["ee.y"])

    def _move_keyboard_z(self, dz: float) -> None:
        self._state["ee.z"] = self._clamp_state("ee.z", self._state["ee.z"] + dz)
        self._motion_baseline["ee.z"] = float(self._state["ee.z"])
        logger.info("Keyboard Z -> ee.z=%.3f", self._state["ee.z"])

    def _adjust_xy_scale(self, delta_m: float) -> None:
        if self._handtracker is not None and hasattr(self._handtracker, "adjust_xy_scale"):
            self._xy_scale = float(self._handtracker.adjust_xy_scale(delta_m))
        else:
            self._xy_scale = max(0.01, min(MAX_XY_SCALE_M, self._xy_scale + delta_m))
        logger.info("Handtracker XY scale -> %.3f m", self._xy_scale)

    def _current_xy_scale(self) -> float:
        if self._handtracker is not None and hasattr(self._handtracker, "get_xy_scale"):
            self._xy_scale = float(self._handtracker.get_xy_scale())
        return self._xy_scale

    def _adjust_z_scale(self, delta_m: float) -> None:
        if self._handtracker is not None and hasattr(self._handtracker, "adjust_z_scale"):
            self._z_scale = float(self._handtracker.adjust_z_scale(delta_m))
        else:
            self._z_scale = max(0.01, min(0.30, self._z_scale + delta_m))
        logger.info("Handtracker Z scale -> %.3f m", self._z_scale)

    def _current_z_scale(self) -> float:
        if self._handtracker is not None and hasattr(self._handtracker, "get_z_scale"):
            self._z_scale = float(self._handtracker.get_z_scale())
        return self._z_scale

    @staticmethod
    def _mapping_status(live: bool, enabled: bool, rebase_pending: bool) -> str:
        if not enabled:
            return "disabled"
        if rebase_pending:
            return "waiting for hand"
        return "live" if live else "held"

    def _normalize_key_char(self, key) -> str | None:
        char = getattr(key, "char", None)
        if char is not None:
            return str(char)

        key_str = str(key)
        match = re.search(r"char='(.+)'", key_str)
        if match is not None:
            return match.group(1)

        return None

    def _read_relative_xy(self) -> dict[str, float] | None:
        if self._handtracker is None or not hasattr(self._handtracker, "get_latest_xy"):
            return None
        try:
            latest_xy = self._handtracker.get_latest_xy()
            top = latest_xy.get("top") if isinstance(latest_xy, dict) else None
            if top is None:
                now = time.monotonic()
                if now - self._last_xy_valid_t <= self._xy_hold_timeout_s:
                    return self._last_relative_xy
                return None
            self._last_relative_xy = {
                "dx_px": float(top.get("dx_px", 0.0)),
                "dy_px": float(top.get("dy_px", 0.0)),
                "dx_norm": float(top.get("dx_norm", 0.0)),
                "dy_norm": float(top.get("dy_norm", 0.0)),
            }
            self._last_xy_valid_t = time.monotonic()
            return self._last_relative_xy
        except Exception:
            logger.exception("Failed to read handtracker relative XY")
            return None

    def _read_relative_z(self) -> float | None:
        if self._handtracker is None or not hasattr(self._handtracker, "get_latest_xy"):
            return None
        try:
            side = self._handtracker.get_latest_xy().get("side")
            now = time.monotonic()
            if side is None:
                if now - self._last_z_valid_t <= self._xy_hold_timeout_s:
                    return self._last_relative_z
                return None
            self._last_relative_z = float(side.get("dz_norm", 0.0))
            self._last_z_valid_t = now
            return self._last_relative_z
        except Exception:
            logger.exception("Failed to read handtracker relative Z")
            return None

    def _calibrate_hand_reference(self, xy: bool = True, z: bool = True) -> bool:
        """Reset the hand origin (top cam for XY, side cam for Z) to the current hand.

        Returns True only if every requested axis group was calibrated, i.e. the
        corresponding hand was detected right now.
        """
        if self._handtracker is None:
            return False
        ok = True
        try:
            latest = self._handtracker.get_latest_xy()
            if xy:
                top = latest.get("top")
                if top is not None and hasattr(self._handtracker, "calibrate_top_xy"):
                    self._handtracker.calibrate_top_xy((top.get("mid_x", 0.0), top.get("mid_y", 0.0)))
                    logger.info("Calibrated top XY midpoint from current hand pose")
                else:
                    ok = False
            if z:
                side = latest.get("side")
                if side is not None and hasattr(self._handtracker, "calibrate_side_z"):
                    self._handtracker.calibrate_side_z((side.get("mid_x", 0.0), side.get("mid_y", 0.0)))
                    logger.info("Calibrated side Z midpoint from current hand pose")
                else:
                    ok = False
        except Exception:
            logger.exception("Failed to calibrate hand reference")
            return False
        return ok

    def _try_rebase_xy(self) -> bool:
        if not self._calibrate_hand_reference(xy=True, z=False):
            return False
        self._motion_baseline["ee.x"] = float(self._state["ee.x"])
        self._motion_baseline["ee.y"] = float(self._state["ee.y"])
        self._last_relative_xy = {"dx_px": 0.0, "dy_px": 0.0, "dx_norm": 0.0, "dy_norm": 0.0}
        self._last_xy_valid_t = time.monotonic()
        self._xy_rebase_pending = False
        return True

    def _try_rebase_z(self) -> bool:
        if not self._calibrate_hand_reference(xy=False, z=True):
            return False
        self._motion_baseline["ee.z"] = float(self._state["ee.z"])
        self._last_relative_z = 0.0
        self._last_z_valid_t = time.monotonic()
        self._z_rebase_pending = False
        return True

    def _toggle_mapping(self, axes: str) -> None:
        if self._control_mode != CONTROL_MODE_HANDTRACKER:
            logger.info("%s mapping toggle ignored: only available in handtracker modes (2 / 3)", axes)
            return
        if axes == "XY":
            self._xy_mapping_enabled = not self._xy_mapping_enabled
            self._xy_rebase_pending = self._xy_mapping_enabled
            enabled = self._xy_mapping_enabled
        else:
            self._z_mapping_enabled = not self._z_mapping_enabled
            self._z_rebase_pending = self._z_mapping_enabled
            enabled = self._z_mapping_enabled
        logger.info(
            "%s mapping %s%s",
            axes,
            "ENABLED" if enabled else "DISABLED (holding last position)",
            " - waiting for hand to re-base origin" if enabled else "",
        )

    def _set_pitch_mapping_mode(self, mode: str) -> None:
        if self._control_mode != CONTROL_MODE_HANDTRACKER:
            logger.info("Pitch mapping toggle ignored: only available in handtracker modes (2 / 3)")
            return
        if mode not in {PITCH_MODE_OFF, PITCH_MODE_2D, PITCH_MODE_3D}:
            return
        self._pitch_mapping_mode = mode
        if mode == PITCH_MODE_OFF:
            self._pitch_rebase_pending = False
            logger.info("Pitch mapping DISABLED (holding last pitch %.1f deg)", self._pitch_down_cmd_deg)
            return
        self._pitch_rebase_pending = True
        self._pitch_down_baseline_deg = float(self._pitch_down_cmd_deg)
        logger.info("Pitch mapping -> %s (waiting for hand to re-base reference)", mode.upper())

    def _toggle_pitch_mapping_mode(self) -> None:
        if self._pitch_mapping_mode == PITCH_MODE_2D:
            self._set_pitch_mapping_mode(PITCH_MODE_3D)
        elif self._pitch_mapping_mode == PITCH_MODE_3D:
            self._set_pitch_mapping_mode(PITCH_MODE_OFF)
        else:
            self._set_pitch_mapping_mode(PITCH_MODE_2D)

    def _toggle_measure_source(self) -> None:
        if self._control_mode != CONTROL_MODE_HANDTRACKER:
            logger.info("Measurement source toggle ignored: only available in handtracker modes (2 / 3)")
            return
        if self._handtracker is None or not hasattr(self._handtracker, "toggle_measure_source"):
            logger.info("Measurement source toggle unavailable: handtracker module not loaded")
            return
        source = self._handtracker.toggle_measure_source()
        self._xy_rebase_pending = True
        self._z_rebase_pending = True
        logger.info("Measurement source -> %s (re-basing XY and Z)", source)

    def _read_pitch_elevation_deg(self) -> float | None:
        if self._handtracker is None or not hasattr(self._handtracker, "get_latest_xy"):
            return None
        if self._pitch_mapping_mode not in {PITCH_MODE_2D, PITCH_MODE_3D}:
            return None
        key = "pitch_2d_deg" if self._pitch_mapping_mode == PITCH_MODE_2D else "pitch_3d_deg"
        try:
            side = self._handtracker.get_latest_xy().get("side")
            now = time.monotonic()
            if side is None or side.get(key) is None:
                if now - self._last_pitch_valid_t <= self._pitch_hold_timeout_s:
                    return self._last_pitch_measurement_deg
                return None
            value = float(side.get(key, 0.0))
            # Gap since the previous *fresh* reading (not the held value). Used to decide
            # whether the reference must be re-anchored after the hand comes back.
            self._pitch_gap_s = now - self._last_pitch_valid_t if self._last_pitch_valid_t > 0.0 else 0.0
            self._last_pitch_measurement_deg = value
            self._last_pitch_valid_t = now
            return value
        except Exception:
            logger.exception("Failed to read handtracker pitch elevation")
            return None

    def _try_rebase_pitch(self) -> bool:
        elevation = self._read_pitch_elevation_deg()
        if elevation is None:
            return False
        self._pitch_elevation_ref_deg = float(elevation)
        self._pitch_down_baseline_deg = float(self._pitch_down_cmd_deg)
        self._last_pitch_delta_ema_deg = 0.0
        self._pitch_rebase_pending = False
        return True

    @staticmethod
    def _apply_deadzone(value: float, deadzone: float) -> float:
        if abs(value) <= deadzone:
            return 0.0
        return math.copysign(abs(value) - deadzone, value)

    def _current_measure_source(self) -> str:
        if self._handtracker is not None and hasattr(self._handtracker, "get_measure_source"):
            try:
                return str(self._handtracker.get_measure_source())
            except Exception:
                return "pixel"
        return "pixel"

    def _control_status_text(self) -> str:
        mode_label = {
            1: "1 keyboard",
            2: "2 handtracker (lightweight)",
            3: "3 handtracker (full)",
        }.get(self._active_mode, str(self._active_mode))
        return (
            f"mode: {mode_label}\n\n"
            f"XY mapping: {'ENABLED' if self._xy_mapping_enabled else 'DISABLED'}\n\n"
            f"Z mapping: {'ENABLED' if self._z_mapping_enabled else 'DISABLED'}\n\n"
            f"pitch: {self._pitch_mapping_mode}\n\n"
            f"measure: {self._current_measure_source()}\n\n"
            f"xy_scale: {self._xy_scale:.3f} m\n\n"
            f"z_scale: {self._z_scale:.3f} m"
        )

    def _push_control_status(self) -> None:
        if self._handtracker is not None and hasattr(self._handtracker, "set_control_status"):
            self._handtracker.set_control_status(self._control_status_text())

    def _update_handtracker_orientation(self) -> None:
        """Rotate the seeded orientation about a fixed horizontal axis by the pitch delta.

        At ``pitch_cmd == pitch_down_seed`` this reproduces the seed orientation exactly,
        so entering modes 2/3 (or moving the hand sideways) no longer jumps the target.
        """
        if self._pitch_R_base is None or self._pitch_axis_n is None:
            return
        delta_rad = math.radians(self._pitch_down_cmd_deg - self._pitch_down_seed_deg)
        r_target = self._rot_axis_angle(self._pitch_axis_n, delta_rad) @ self._pitch_R_base
        rotvec = Rotation.from_matrix(np.asarray(r_target, dtype=float)).as_rotvec()
        self._orientation_lock = {
            "ee.wx": float(rotvec[0]),
            "ee.wy": float(rotvec[1]),
            "ee.wz": float(rotvec[2]),
        }
        self._apply_orientation_lock()

    def _reset_to_seed(self) -> None:
        """Key x: send the arm back to the seeded start pose and re-anchor the hand there."""
        for key in ("ee.x", "ee.y", "ee.z"):
            self._state[key] = self._clamp_state(key, float(self._seed_position.get(key, self._state[key])))
        self._motion_baseline = {k: float(self._state[k]) for k in ("ee.x", "ee.y", "ee.z")}
        self._orientation_lock = dict(self._seed_orientation_lock)
        self._apply_orientation_lock()
        self._pitch_down_cmd_deg = float(self._pitch_down_seed_deg)
        self._pitch_down_baseline_deg = float(self._pitch_down_seed_deg)
        self._last_pitch_delta_ema_deg = 0.0
        self._pitch_elevation_ref_deg = None
        self._pitch_rebase_pending = self._pitch_mapping_mode != PITCH_MODE_OFF
        # The hand's current pose becomes the new neutral; rebase on the next handtracker tick.
        self._xy_rebase_pending = True
        self._z_rebase_pending = True
        logger.info(
            "Reset to seed pose: ee=(%.3f, %.3f, %.3f), pitch_down=%.1f deg",
            self._state["ee.x"],
            self._state["ee.y"],
            self._state["ee.z"],
            self._pitch_down_cmd_deg,
        )

    def _apply_motion_from_handtracker(self) -> None:
        xy = None
        if self._xy_mapping_enabled:
            if self._xy_rebase_pending:
                if self._try_rebase_xy():
                    logger.info("XY origin re-based at ee.x=%.3f ee.y=%.3f", self._state["ee.x"], self._state["ee.y"])
            else:
                xy = self._read_relative_xy()
        if xy is not None:
            scale = self._current_xy_scale()
            target_x = self._motion_baseline["ee.x"] + xy["dx_norm"] * scale
            target_y = self._motion_baseline["ee.y"] + xy["dy_norm"] * scale
            self._state["ee.x"] = self._clamp_state("ee.x", target_x)
            self._state["ee.y"] = self._clamp_state("ee.y", target_y)

        dz_norm = None
        if self._z_mapping_enabled:
            if self._z_rebase_pending:
                if self._try_rebase_z():
                    logger.info("Z origin re-based at ee.z=%.3f", self._state["ee.z"])
            else:
                dz_norm = self._read_relative_z()
        if dz_norm is not None:
            target_z = self._motion_baseline["ee.z"] + dz_norm * self._current_z_scale()
            self._state["ee.z"] = self._clamp_state("ee.z", target_z)

        # Refresh the pitch tick clock EVERY tick and clamp dt, so the first update after
        # a gap (hand lost, rebase) cannot bypass the rate limit.
        pitch_now = time.monotonic()
        pitch_dt = max(PITCH_DT_MIN_S, min(PITCH_DT_MAX_S, pitch_now - self._last_pitch_cmd_update_t))
        self._last_pitch_cmd_update_t = pitch_now

        pitch_live = False
        if self._pitch_mapping_mode != PITCH_MODE_OFF:
            if self._pitch_rebase_pending:
                if self._try_rebase_pitch():
                    logger.info(
                        "Pitch reference re-based at %.2f deg (pitch_down=%.1f deg)",
                        float(self._pitch_elevation_ref_deg or 0.0),
                        self._pitch_down_cmd_deg,
                    )
            else:
                elevation = self._read_pitch_elevation_deg()
                if elevation is not None:
                    if self._pitch_elevation_ref_deg is None or self._pitch_gap_s > PITCH_REBASE_TIMEOUT_S:
                        # Hand was gone long enough: re-anchor to the current elevation.
                        self._pitch_elevation_ref_deg = float(elevation)
                        self._pitch_down_baseline_deg = float(self._pitch_down_cmd_deg)
                        self._last_pitch_delta_ema_deg = 0.0
                        logger.info(
                            "Pitch reference auto-re-based after %.2fs gap at %.2f deg",
                            self._pitch_gap_s,
                            float(elevation),
                        )
                    else:
                        hand_delta = float(elevation - self._pitch_elevation_ref_deg)
                        hand_delta = self._apply_deadzone(hand_delta, PITCH_DEADZONE_DEG)
                        hand_delta = max(-PITCH_CLAMP_DEG, min(PITCH_CLAMP_DEG, hand_delta))
                        # Time-based smoothing so the response speed does not depend on loop rate.
                        alpha = 1.0 - math.exp(-pitch_dt / PITCH_EMA_TAU_S)
                        self._last_pitch_delta_ema_deg = (
                            alpha * hand_delta + (1.0 - alpha) * self._last_pitch_delta_ema_deg
                        )
                        pitch_target = self._pitch_down_baseline_deg - self._pitch_gain * self._last_pitch_delta_ema_deg
                        pitch_target = max(PITCH_DOWN_MIN_DEG, min(PITCH_DOWN_MAX_DEG, pitch_target))

                        max_step = PITCH_RATE_LIMIT_DEG_PER_S * pitch_dt
                        delta_cmd = pitch_target - self._pitch_down_cmd_deg
                        if abs(delta_cmd) > max_step:
                            pitch_target = self._pitch_down_cmd_deg + math.copysign(max_step, delta_cmd)
                        self._pitch_down_cmd_deg = float(pitch_target)
                        pitch_live = True

        self._update_handtracker_orientation()

        now = time.monotonic()
        if now - self._last_hand_log_t >= self._hand_log_interval_s:
            self._last_hand_log_t = now
            logger.info(
                "Hand mapped -> ee.x=%.3f ee.y=%.3f ee.z=%.3f pitch_down=%.1f deg (xy %s, z %s, pitch %s:%s)",
                self._state["ee.x"],
                self._state["ee.y"],
                self._state["ee.z"],
                self._pitch_down_cmd_deg,
                self._mapping_status(xy is not None, self._xy_mapping_enabled, self._xy_rebase_pending),
                self._mapping_status(dz_norm is not None, self._z_mapping_enabled, self._z_rebase_pending),
                self._pitch_mapping_mode,
                "live" if pitch_live else ("waiting" if self._pitch_rebase_pending else "held"),
            )

    def get_action(self) -> dict[str, float]:
        """Return the current end-effector pose."""
        self._apply_orientation_lock()
        if self._control_mode == CONTROL_MODE_HANDTRACKER and self._handtracker is not None and hasattr(self._handtracker, "get_gripper_fraction"):
            try:
                frac = float(self._handtracker.get_gripper_fraction())
                self._state[EE_GRIPPER_KEY] = self._clamp_state(EE_GRIPPER_KEY, frac * 100.0)
            except Exception:
                logger.exception("handtracker.get_gripper_fraction() failed; using keyboard state")

        if self._control_mode == CONTROL_MODE_HANDTRACKER:
            self._apply_motion_from_handtracker()

        self._push_control_status()

        return {key: float(self._state[key]) for key in self.motors}

    def connect(self) -> None:
        from pynput import keyboard

        def on_press(key) -> None:
            try:
                lowered_char = self._normalize_key_char(key)
                if lowered_char is not None:
                    lowered = lowered_char.lower()
                    self._last_key = lowered

                    if lowered == "1":
                        self._active_mode = 1
                        self._set_control_mode(CONTROL_MODE_KEYBOARD)
                        self._set_handtracker_logging_mode("off")
                        return
                    if lowered == "2":
                        self._active_mode = 2
                        self._set_control_mode(CONTROL_MODE_HANDTRACKER)
                        self._set_handtracker_logging_mode("lightweight")
                        return
                    if lowered == "3":
                        self._active_mode = 3
                        self._set_control_mode(CONTROL_MODE_HANDTRACKER)
                        self._set_handtracker_logging_mode("full")
                        return
                    if lowered == "4":
                        self._toggle_mapping("XY")
                        return
                    if lowered == "5":
                        self._toggle_mapping("Z")
                        return
                    if lowered == "6":
                        self._toggle_pitch_mapping_mode()
                        return
                    if lowered == "7":
                        self._toggle_measure_source()
                        return

                    if lowered == "w":
                        self._move_keyboard_z(self._keyboard_step)
                    elif lowered == "s":
                        self._move_keyboard_z(-self._keyboard_step)
                    elif lowered == "o" and self._handtracker is not None and hasattr(self._handtracker, "calibrate_open"):
                        self._handtracker.calibrate_open()
                    elif lowered == "c" and self._handtracker is not None and hasattr(self._handtracker, "calibrate_closed"):
                        self._handtracker.calibrate_closed()
                    elif lowered == "x":
                        self._reset_to_seed()
                    elif lowered == "q" and self._control_mode == CONTROL_MODE_HANDTRACKER:
                        self._adjust_xy_scale(-XY_SCALE_STEP_M)
                        return
                    elif lowered == "e" and self._control_mode == CONTROL_MODE_HANDTRACKER:
                        self._adjust_xy_scale(XY_SCALE_STEP_M)
                        return
                    elif lowered == "," and self._control_mode == CONTROL_MODE_HANDTRACKER:
                        self._adjust_z_scale(-Z_SCALE_STEP_M)
                    elif lowered == "." and self._control_mode == CONTROL_MODE_HANDTRACKER:
                        self._adjust_z_scale(Z_SCALE_STEP_M)
                    elif lowered == "]" and self._handtracker is not None and hasattr(self._handtracker, "adjust_smoothing"):
                        self._handtracker.adjust_smoothing(0.05)
                    elif lowered == "[" and self._handtracker is not None and hasattr(self._handtracker, "adjust_smoothing"):
                        self._handtracker.adjust_smoothing(-0.05)
                    elif lowered == "r":
                        self._reset_pose()
                    elif self._control_mode == CONTROL_MODE_KEYBOARD:
                        if lowered == "a":
                            self._move_keyboard_xy(-self._keyboard_step, 0.0)
                        elif lowered == "d":
                            self._move_keyboard_xy(self._keyboard_step, 0.0)
                        elif lowered == "q":
                            self._state[EE_GRIPPER_KEY] = self._clamp_state(EE_GRIPPER_KEY, self._state[EE_GRIPPER_KEY] - 5.0)
                            logger.info("Keyboard gripper -> %.2f", self._state[EE_GRIPPER_KEY])
                        elif lowered == "e":
                            self._state[EE_GRIPPER_KEY] = self._clamp_state(EE_GRIPPER_KEY, self._state[EE_GRIPPER_KEY] + 5.0)
                            logger.info("Keyboard gripper -> %.2f", self._state[EE_GRIPPER_KEY])
                elif key == keyboard.Key.up:
                    if self._control_mode == CONTROL_MODE_KEYBOARD:
                        self._move_keyboard_xy(0.0, self._keyboard_step)
                elif key == keyboard.Key.down:
                    if self._control_mode == CONTROL_MODE_KEYBOARD:
                        self._move_keyboard_xy(0.0, -self._keyboard_step)
                elif key == keyboard.Key.left:
                    if self._control_mode == CONTROL_MODE_KEYBOARD:
                        self._move_keyboard_xy(-self._keyboard_step, 0.0)
                elif key == keyboard.Key.right:
                    if self._control_mode == CONTROL_MODE_KEYBOARD:
                        self._move_keyboard_xy(self._keyboard_step, 0.0)
            except Exception:
                logger.exception("Error handling key press")

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Optionally keep internal state in sync from feedback if needed
        pass

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
