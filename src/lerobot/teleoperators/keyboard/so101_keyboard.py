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
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from lerobot.utils.import_utils import _pynput_available, require_package
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


@TeleoperatorConfig.register_subclass("keyboard_so_leader")
@dataclass
class KeyboardSOLeaderConfig(TeleoperatorConfig):
    """Config for the keyboard SO leader teleoperator."""

    step_deg: float = 1.0
    use_degrees: bool = True
    handtracker_module: str | None = None


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
        self._xy_scale: float = 0.05
        self._keyboard_step: float = self.config.step_deg * 0.01
        self._last_relative_xy: dict[str, float] = {"dx_px": 0.0, "dy_px": 0.0, "dx_norm": 0.0, "dy_norm": 0.0}
        self._last_xy_valid_t: float = 0.0
        self._last_relative_z: float = 0.0
        self._last_z_valid_t: float = 0.0
        self._z_scale: float = 0.08
        self._xy_hold_timeout_s: float = 0.35
        self._last_hand_log_t: float = 0.0
        self._hand_log_interval_s: float = 0.5
        self._orientation_lock = dict(LOCKED_ORIENTATION)
        self._apply_orientation_lock()

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
            self._calibrate_top_xy_from_current_hand()
        self._control_mode = mode
        logger.info("Control mode switched to: %s", self._control_mode)

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
        self._orientation_lock = {
            "ee.wx": float(self._state["ee.wx"]),
            "ee.wy": float(self._state["ee.wy"]),
            "ee.wz": float(self._state["ee.wz"]),
        }
        self._apply_orientation_lock()
        logger.info("Seeded initial EE pose: %s", {k: self._state[k] for k in self.motors})

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

    def _calibrate_top_xy_from_current_hand(self) -> bool:
        """Reset the hand reference points (top cam for XY, side cam for Z) to the current hand."""
        if self._handtracker is None:
            return False
        calibrated = False
        try:
            latest = self._handtracker.get_latest_xy()
            top = latest.get("top")
            if top is not None and hasattr(self._handtracker, "calibrate_top_xy"):
                self._handtracker.calibrate_top_xy((top.get("mid_x", 0.0), top.get("mid_y", 0.0)))
                logger.info("Calibrated top XY midpoint from current hand pose")
                calibrated = True
            side = latest.get("side")
            if side is not None and hasattr(self._handtracker, "calibrate_side_z"):
                self._handtracker.calibrate_side_z((side.get("mid_x", 0.0), side.get("mid_y", 0.0)))
                logger.info("Calibrated side Z midpoint from current hand pose")
                calibrated = True
        except Exception:
            logger.exception("Failed to calibrate hand reference")
        return calibrated

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

    def _apply_motion_from_handtracker(self) -> None:
        xy = self._read_relative_xy()
        if xy is not None:
            scale = self._current_xy_scale()
            target_x = self._motion_baseline["ee.x"] + xy["dx_norm"] * scale
            target_y = self._motion_baseline["ee.y"] + xy["dy_norm"] * scale
            self._state["ee.x"] = self._clamp_state("ee.x", target_x)
            self._state["ee.y"] = self._clamp_state("ee.y", target_y)

        dz_norm = self._read_relative_z()
        if dz_norm is not None:
            target_z = self._motion_baseline["ee.z"] + dz_norm * self._current_z_scale()
            self._state["ee.z"] = self._clamp_state("ee.z", target_z)

        now = time.monotonic()
        if now - self._last_hand_log_t >= self._hand_log_interval_s:
            self._last_hand_log_t = now
            logger.info(
                "Hand mapped -> ee.x=%.3f ee.y=%.3f ee.z=%.3f (xy %s, z %s)",
                self._state["ee.x"],
                self._state["ee.y"],
                self._state["ee.z"],
                "live" if xy is not None else "held",
                "live" if dz_norm is not None else "held",
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

                    if lowered == "w":
                        self._move_keyboard_z(self._keyboard_step)
                    elif lowered == "s":
                        self._move_keyboard_z(-self._keyboard_step)
                    elif lowered == "o" and self._handtracker is not None and hasattr(self._handtracker, "calibrate_open"):
                        self._handtracker.calibrate_open()
                    elif lowered == "c" and self._handtracker is not None and hasattr(self._handtracker, "calibrate_closed"):
                        self._handtracker.calibrate_closed()
                    elif lowered == "x":
                        self._calibrate_top_xy_from_current_hand()
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
