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
XY_SCALE_STEP_M = 0.005
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
PITCH_EMA_ALPHA = 0.25
PITCH_RATE_LIMIT_DEG_PER_S = 60.0
PITCH_GAIN_DEFAULT = 1.0
PITCH_DOWN_MIN_DEG = 0.0
PITCH_DOWN_MAX_DEG = 90.0
PITCH_DOWN_DEFAULT_DEG = 45.0


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
        self._last_env_yaw_rad: float = 0.0

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
        if self._handtracker is None:
            return
        enabled = str(mode).lower() != "off"
        if hasattr(self._handtracker, "enable_rerun_logging"):
            try:
                self._handtracker.enable_rerun_logging(enabled, mode)
            except TypeError:
                self._handtracker.enable_rerun_logging(enabled)
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
    def _rot_z(theta_rad: float) -> np.ndarray:
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    @staticmethod
    def _rot_y(theta_rad: float) -> np.ndarray:
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)

    def _compose_orientation(self, yaw_rad: float, pitch_down_deg: float, local_roll_deg: float = 0.0) -> dict[str, float]:
        # R_target = Rz(yaw) * Ry(90deg + pitch_down) * Rz_local(roll)
        r_target = (
            self._rot_z(yaw_rad)
            @ self._rot_y(math.radians(90.0 + float(pitch_down_deg)))
            @ self._rot_z(math.radians(float(local_roll_deg)))
        )
        rotvec = Rotation.from_matrix(r_target).as_rotvec()
        return {"ee.wx": float(rotvec[0]), "ee.wy": float(rotvec[1]), "ee.wz": float(rotvec[2])}

    def _compute_env_yaw_rad(self) -> float:
        # User frame -> env frame mapping (same transform as run_sim_keyboard adapter)
        # env_x = user_y, env_y = -user_x
        user_x = float(self._state["ee.x"])
        user_y = float(self._state["ee.y"])
        env_x = user_y
        env_y = -user_x
        if abs(env_x) < 1e-6 and abs(env_y) < 1e-6:
            return self._last_env_yaw_rad
        yaw = math.atan2(env_y, env_x)
        self._last_env_yaw_rad = yaw
        return yaw

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
        self._orientation_lock = self._rolled_orientation(
            (float(self._state["ee.wx"]), float(self._state["ee.wy"]), float(self._state["ee.wz"])),
            self.config.wrist_roll_offset_deg,
        )
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
            self._xy_scale = max(0.01, min(0.20, self._xy_scale + delta_m))
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

    def _update_handtracker_orientation(self) -> None:
        yaw = self._compute_env_yaw_rad()
        orientation = self._compose_orientation(yaw, self._pitch_down_cmd_deg, local_roll_deg=0.0)
        self._orientation_lock = orientation
        self._apply_orientation_lock()

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
                if elevation is not None and self._pitch_elevation_ref_deg is not None:
                    hand_delta = float(elevation - self._pitch_elevation_ref_deg)
                    hand_delta = self._apply_deadzone(hand_delta, PITCH_DEADZONE_DEG)
                    hand_delta = max(-PITCH_CLAMP_DEG, min(PITCH_CLAMP_DEG, hand_delta))
                    self._last_pitch_delta_ema_deg = (
                        PITCH_EMA_ALPHA * hand_delta + (1.0 - PITCH_EMA_ALPHA) * self._last_pitch_delta_ema_deg
                    )
                    pitch_target = self._pitch_down_baseline_deg - self._pitch_gain * self._last_pitch_delta_ema_deg
                    pitch_target = max(PITCH_DOWN_MIN_DEG, min(PITCH_DOWN_MAX_DEG, pitch_target))

                    now = time.monotonic()
                    dt = max(1e-3, now - self._last_pitch_cmd_update_t)
                    max_step = PITCH_RATE_LIMIT_DEG_PER_S * dt
                    delta_cmd = pitch_target - self._pitch_down_cmd_deg
                    if abs(delta_cmd) > max_step:
                        pitch_target = self._pitch_down_cmd_deg + math.copysign(max_step, delta_cmd)
                    self._pitch_down_cmd_deg = float(pitch_target)
                    self._last_pitch_cmd_update_t = now
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

        return {key: float(self._state[key]) for key in self.motors}

    def connect(self) -> None:
        from pynput import keyboard

        def on_press(key) -> None:
            try:
                lowered_char = self._normalize_key_char(key)
                if lowered_char is not None:
                    lowered = lowered_char.lower()

                    if lowered == "1":
                        self._set_control_mode(CONTROL_MODE_KEYBOARD)
                        self._set_handtracker_logging_mode("off")
                        return
                    if lowered == "2":
                        self._set_control_mode(CONTROL_MODE_HANDTRACKER)
                        self._set_handtracker_logging_mode("lightweight")
                        return
                    if lowered == "3":
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

                    if lowered == "w":
                        self._move_keyboard_z(self._keyboard_step)
                    elif lowered == "s":
                        self._move_keyboard_z(-self._keyboard_step)
                    elif lowered == "o" and self._handtracker is not None and hasattr(self._handtracker, "calibrate_open"):
                        self._handtracker.calibrate_open()
                    elif lowered == "c" and self._handtracker is not None and hasattr(self._handtracker, "calibrate_closed"):
                        self._handtracker.calibrate_closed()
                    elif lowered == "x":
                        self._calibrate_hand_reference()
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
