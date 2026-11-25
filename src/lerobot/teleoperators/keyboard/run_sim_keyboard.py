#!/usr/bin/env python3
"""Run keyboard teleoperation against the SO101 simulator follower.

Example:
    cd "/Users/hoihungkobehon/hand tracker project"
    source lerobot-sim/.venv/bin/activate
    export PYTHONPATH="lerobot-sim/src:so101-nexus/src"
    python lerobot-sim/src/lerobot/teleoperators/keyboard/run_sim_keyboard.py
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from pprint import pformat

import numpy as np

logger = logging.getLogger(__name__)

# Warn when the commanded orientation changes by more than this between two ticks.
ORIENTATION_JUMP_WARN_DEG = 15.0
ORIENTATION_JUMP_WARN_COOLDOWN_S = 0.5
# Safety net: if the sim's observed orientation stays this far from the commanded
# orientation for this many consecutive frames, the IK has dropped into a mirrored
# solution. Return to the seed pose and re-anchor the hand.
ORIENTATION_RECOVER_DEG = 90.0
ORIENTATION_RECOVER_FRAMES = 10
ORIENTATION_RECOVER_COOLDOWN_S = 3.0
# Periodic text log of the target/observed orientation so the cause of a "points at
# the ceiling" event is visible without opening Rerun.
DIAGNOSTICS_LOG_INTERVAL_S = 2.0
# Seed the arm at env x ~= 0.30 (user-frame y) so there is room on both sides before
# the unreachable region near the base (env x < 0.20).
SEED_ENV_X = 0.30
SEED_WARMUP_STEPS = 35
SEED_WARMUP_STABLE_STEPS = 5
SEED_WARMUP_TOLERANCE_M = 0.01
SEED_WARMUP_ROT_TOLERANCE_DEG = 1.0


def _rotvec_from_action(action: dict) -> np.ndarray | None:
    if not all(k in action for k in ("ee.wx", "ee.wy", "ee.wz")):
        return None
    return np.array([float(action["ee.wx"]), float(action["ee.wy"]), float(action["ee.wz"])])


def _tool_pitch_down_deg(rotvec: np.ndarray) -> float:
    """Pitch of the tool z axis below horizontal, in degrees (0 = level, 90 = straight down)."""
    # Imported lazily: this script adds its own sys.path entries inside main().
    from lerobot.utils.rotation import Rotation

    tool_z = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()[:, 2]
    return math.degrees(math.atan2(-tool_z[2], math.hypot(tool_z[0], tool_z[1])))


def _rotation_delta_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    from lerobot.utils.rotation import Rotation

    delta = Rotation.from_rotvec(np.asarray(r1, dtype=float)).inv() * Rotation.from_rotvec(
        np.asarray(r2, dtype=float)
    )
    return math.degrees(float(np.linalg.norm(delta.as_rotvec())))


def _warmup_to_seed_position(
    robot, initial_pose: dict[str, float], gripper_value: float
) -> dict[str, float]:
    """Move the arm to the seed position (env x ~= SEED_ENV_X) and let it settle.

    The sim starts with the arm close to the base (env x ~= 0.225), where the IK can
    saturate and flip the gripper. Moving forward first gives the hand tracker room on
    both sides. Returns the observed pose after settling (falls back to ``initial_pose``
    if the FK is unavailable).
    """
    action = {
        "ee.x": 0.0,  # user x -> env y
        "ee.y": SEED_ENV_X,  # user y -> env x
        "ee.z": float(initial_pose.get("ee.z", 0.15)),
        "ee.wx": float(initial_pose.get("ee.wx", 0.0)),
        "ee.wy": float(initial_pose.get("ee.wy", 0.0)),
        "ee.wz": float(initial_pose.get("ee.wz", 0.0)),
        "ee.gripper_pos": float(gripper_value),
    }
    observed = dict(initial_pose)
    previous = dict(initial_pose)
    stable = 0
    for _ in range(SEED_WARMUP_STEPS):
        robot.send_action(action)
        observation = robot.get_observation()
        pose = robot.get_observed_ee_pose(observation)
        if not pose:
            return previous
        observed = pose
        pos_err = abs(float(pose.get("ee.y", 0.0)) - SEED_ENV_X)
        rot_err = _rotation_delta_deg(
            [previous.get("ee.wx", 0.0), previous.get("ee.wy", 0.0), previous.get("ee.wz", 0.0)],
            [pose.get("ee.wx", 0.0), pose.get("ee.wy", 0.0), pose.get("ee.wz", 0.0)],
        )
        previous = dict(pose)
        # The arm can pass through a momentarily "stable" transient on its way to the
        # real steady state (position converges before orientation), so require several
        # consecutive stable steps before stopping.
        if pos_err < SEED_WARMUP_TOLERANCE_M and rot_err < SEED_WARMUP_ROT_TOLERANCE_DEG:
            stable += 1
            if stable >= SEED_WARMUP_STABLE_STEPS:
                break
        else:
            stable = 0
    return observed


class _RerunEeLoggingRobot:
    def __init__(self, robot, enabled: bool) -> None:
        self._robot = robot
        self._enabled = enabled
        self._last_observation = None
        # Keyboard/user frame uses Y as "forward on overhead".
        # Env EE frame in this sim appears swapped in XY from overhead view.
        # Remap at the adapter boundary so target/observed semantics match.
        self._swap_xy_user_env = True
        self._compute_forward_kinematics_joints_to_ee = None
        self._kinematics = None
        self._motor_names: list[str] = []
        self._teleop = None
        self._last_target_rotvec: np.ndarray | None = None
        self._last_orient_warn_t = 0.0
        # Orientation watchdog state (runs even when Rerun logging is disabled).
        self._watchdog_enabled = False
        self._orient_bad_frames = 0
        self._last_recover_t = 0.0
        self._last_diag_log_t = 0.0

        # Build FK/kinematics UNCONDITIONALLY so seed_initial_pose and the error
        # diagnostics work even with --no-display-data. Only Rerun logging is gated
        # by `enabled`.
        try:
            from lerobot.model import RobotKinematics
            from lerobot.robots.so_follower.robot_kinematic_processor import compute_forward_kinematics_joints_to_ee
            from so101_nexus import get_so101_urdf_path  # type: ignore[import-not-found]
            from so101_nexus.config import SO101_TCP_FRAME_NAME  # type: ignore[import-not-found]

            self._compute_forward_kinematics_joints_to_ee = compute_forward_kinematics_joints_to_ee
            self._kinematics = RobotKinematics(str(get_so101_urdf_path()), SO101_TCP_FRAME_NAME)
            self._motor_names = list(getattr(robot, "motors", []))
        except Exception:
            logger.exception("Could not build FK/kinematics; observed-pose diagnostics unavailable")

    def set_teleop(self, teleop) -> None:
        """Give the logger access to the teleop so warnings can name the active keys/state."""
        self._teleop = teleop

    def enable_watchdog(self) -> None:
        """Start the orientation watchdog once the teleop has been seeded."""
        self._watchdog_enabled = True
        self._last_diag_log_t = time.monotonic()

    def _overlay_overhead_grid(self, observation: dict) -> None:
        frame = observation.get("overhead")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return

        try:
            import cv2
        except Exception:
            return

        canvas = frame.copy()
        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        grid_step = max(20, min(h, w) // 8)
        grid_color = (80, 80, 80)
        axis_x_color = (255, 120, 120)
        axis_y_color = (120, 255, 120)
        origin_color = (255, 255, 0)

        for x in range(cx % grid_step, w, grid_step):
            cv2.line(canvas, (x, 0), (x, h - 1), grid_color, 1)
        for y in range(cy % grid_step, h, grid_step):
            cv2.line(canvas, (0, y), (w - 1, y), grid_color, 1)

        cv2.line(canvas, (0, cy), (w - 1, cy), axis_x_color, 2)
        cv2.line(canvas, (cx, 0), (cx, h - 1), axis_y_color, 2)
        cv2.circle(canvas, (cx, cy), 5, origin_color, -1)
        cv2.putText(canvas, "image center", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, origin_color, 1)
        cv2.putText(canvas, "+X", (w - 38, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_x_color, 2)
        cv2.putText(canvas, "+Y", (cx + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_y_color, 2)

        observation["overhead"] = canvas

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def connect(self) -> None:
        return self._robot.connect()

    def disconnect(self) -> None:
        return self._robot.disconnect()

    def _log_pose(self, entity_path: str, pose: dict[str, float]) -> None:
        if not self._enabled:
            return

        try:
            import rerun as rr
        except Exception:
            return

        for axis in ("x", "y", "z"):
            value = pose.get(axis)
            if value is not None:
                rr.log(f"{entity_path}/{axis}", rr.Scalars(float(value)))

    def _target_pose(self, action: dict[str, float]) -> dict[str, float]:
        return {
            "x": float(action.get("ee.x", 0.0)),
            "y": float(action.get("ee.y", 0.0)),
            "z": float(action.get("ee.z", 0.0)),
        }

    def _user_to_env_action(self, action: dict[str, float]) -> dict[str, float]:
        mapped = dict(action)
        if self._swap_xy_user_env and "ee.x" in mapped and "ee.y" in mapped:
            # user frame: (+y forward in overhead, +x right)
            # env frame: axes are swapped and env-y has opposite sign vs user-x
            mapped["ee.x"], mapped["ee.y"] = float(mapped["ee.y"]), float(-mapped["ee.x"])
        return mapped

    def _env_to_user_xy(self, x: float, y: float) -> tuple[float, float]:
        if self._swap_xy_user_env:
            return float(-y), float(x)
        return float(x), float(y)

    def _observed_pose(self, observation: dict[str, float]) -> dict[str, float]:
        if self._compute_forward_kinematics_joints_to_ee is None or self._kinematics is None:
            return {}

        ee_observation = self._compute_forward_kinematics_joints_to_ee(
            dict(observation), self._kinematics, self._motor_names
        )
        x_user, y_user = self._env_to_user_xy(
            float(ee_observation.get("ee.x", 0.0)),
            float(ee_observation.get("ee.y", 0.0)),
        )
        return {
            "x": x_user,
            "y": y_user,
            "z": float(ee_observation.get("ee.z", 0.0)),
        }

    def get_observed_ee_pose(self, observation: dict[str, float]) -> dict[str, float]:
        if self._compute_forward_kinematics_joints_to_ee is None or self._kinematics is None:
            return {}

        ee_observation = self._compute_forward_kinematics_joints_to_ee(
            dict(observation), self._kinematics, self._motor_names
        )
        x_user, y_user = self._env_to_user_xy(
            float(ee_observation.get("ee.x", 0.0)),
            float(ee_observation.get("ee.y", 0.0)),
        )
        return {
            "ee.x": x_user,
            "ee.y": y_user,
            "ee.z": float(ee_observation.get("ee.z", 0.0)),
            "ee.wx": float(ee_observation.get("ee.wx", 0.0)),
            "ee.wy": float(ee_observation.get("ee.wy", 0.0)),
            "ee.wz": float(ee_observation.get("ee.wz", 0.0)),
        }

    def get_observation(self):
        observation = self._robot.get_observation()
        # The grid overlay is a visual aid only; skip it when not displaying.
        if self._enabled:
            self._overlay_overhead_grid(observation)
        self._last_observation = observation
        return observation

    def _observed_env_pose(self, observation: dict[str, float]) -> dict[str, float]:
        """FK of the current joints in the env (sim) frame, including orientation."""
        if self._compute_forward_kinematics_joints_to_ee is None or self._kinematics is None:
            return {}
        return self._compute_forward_kinematics_joints_to_ee(
            dict(observation), self._kinematics, self._motor_names
        )

    def _teleop_state_text(self) -> str:
        teleop = self._teleop
        if teleop is None:
            return "teleop=?"
        return (
            f"last_key={getattr(teleop, '_last_key', None)}, "
            f"control_mode={getattr(teleop, '_control_mode', '?')}, "
            f"pitch_mode={getattr(teleop, '_pitch_mapping_mode', '?')}, "
            f"pitch_cmd={float(getattr(teleop, '_pitch_down_cmd_deg', 0.0)):.1f}deg"
        )

    def _log_diagnostics(self, action: dict[str, float], target_rotvec: np.ndarray | None) -> None:
        # Target orientation jump warning: always on, so it also helps with --no-display-data.
        if target_rotvec is not None:
            if self._last_target_rotvec is not None:
                delta_deg = _rotation_delta_deg(self._last_target_rotvec, target_rotvec)
                now = time.monotonic()
                if delta_deg > ORIENTATION_JUMP_WARN_DEG and now - self._last_orient_warn_t > ORIENTATION_JUMP_WARN_COOLDOWN_S:
                    self._last_orient_warn_t = now
                    logger.warning("Target orientation jumped %.1f deg (%s)", delta_deg, self._teleop_state_text())
            self._last_target_rotvec = target_rotvec

        if not self._enabled or self._last_observation is None:
            return
        try:
            import rerun as rr
        except Exception:
            return

        try:
            env_target = self._user_to_env_action(action)
            observed = self._observed_env_pose(self._last_observation)
            if not observed:
                return

            pos_err_mm = 1000.0 * math.sqrt(
                (float(env_target.get("ee.x", 0.0)) - float(observed.get("ee.x", 0.0))) ** 2
                + (float(env_target.get("ee.y", 0.0)) - float(observed.get("ee.y", 0.0))) ** 2
                + (float(env_target.get("ee.z", 0.0)) - float(observed.get("ee.z", 0.0))) ** 2
            )
            rr.log("teleop/position_error_mm", rr.Scalars(pos_err_mm))

            obs_rotvec = np.array(
                [float(observed.get("ee.wx", 0.0)), float(observed.get("ee.wy", 0.0)), float(observed.get("ee.wz", 0.0))]
            )
            rr.log("teleop/observed_pitch_down_deg", rr.Scalars(_tool_pitch_down_deg(obs_rotvec)))
            if target_rotvec is not None:
                rr.log("teleop/target_pitch_down_deg", rr.Scalars(_tool_pitch_down_deg(target_rotvec)))
                rr.log("teleop/orientation_error_deg", rr.Scalars(_rotation_delta_deg(target_rotvec, obs_rotvec)))
        except Exception:
            logger.exception("Failed to log EE diagnostics")

    def _check_orientation_watchdog(self, action: dict[str, float], target_rotvec: np.ndarray | None) -> None:
        """Log target/observed orientation and recover from a mirrored IK solution.

        Runs on every send_action, even when Rerun logging is disabled, so a "points at
        the ceiling" event is visible in the text log.
        """
        if target_rotvec is None or self._last_observation is None:
            return
        if not self._watchdog_enabled:
            return
        observed = self._observed_env_pose(self._last_observation)
        if not observed:
            return

        obs_rotvec = np.array(
            [
                float(observed.get("ee.wx", 0.0)),
                float(observed.get("ee.wy", 0.0)),
                float(observed.get("ee.wz", 0.0)),
            ]
        )
        target_pitch = _tool_pitch_down_deg(target_rotvec)
        observed_pitch = _tool_pitch_down_deg(obs_rotvec)
        orient_err = _rotation_delta_deg(target_rotvec, obs_rotvec)

        env_target = self._user_to_env_action(action)
        pos_err_mm = 1000.0 * math.sqrt(
            (float(env_target.get("ee.x", 0.0)) - float(observed.get("ee.x", 0.0))) ** 2
            + (float(env_target.get("ee.y", 0.0)) - float(observed.get("ee.y", 0.0))) ** 2
            + (float(env_target.get("ee.z", 0.0)) - float(observed.get("ee.z", 0.0))) ** 2
        )

        now = time.monotonic()
        if now - self._last_diag_log_t >= DIAGNOSTICS_LOG_INTERVAL_S:
            self._last_diag_log_t = now
            logger.info(
                "EE diag: target_pitch=%.1fdeg observed_pitch=%.1fdeg orient_err=%.1fdeg "
                "pos_err=%.1fmm (%s)",
                target_pitch,
                observed_pitch,
                orient_err,
                pos_err_mm,
                self._teleop_state_text(),
            )

        # Recovery: only when the orientation error is large and sustained.
        if orient_err > ORIENTATION_RECOVER_DEG:
            self._orient_bad_frames += 1
        else:
            self._orient_bad_frames = 0

        if (
            self._orient_bad_frames >= ORIENTATION_RECOVER_FRAMES
            and now - self._last_recover_t >= ORIENTATION_RECOVER_COOLDOWN_S
            and self._teleop is not None
            and hasattr(self._teleop, "recover_orientation")
        ):
            self._last_recover_t = now
            self._orient_bad_frames = 0
            logger.warning(
                "Orientation watchdog: observed orientation stayed %.1fdeg from target "
                "(target_pitch=%.1fdeg observed_pitch=%.1fdeg, pos_err=%.1fmm); "
                "returning to seed pose (%s)",
                orient_err,
                target_pitch,
                observed_pitch,
                pos_err_mm,
                self._teleop_state_text(),
            )
            try:
                self._teleop.recover_orientation(reason="orientation watchdog")
            except Exception:
                logger.exception("Orientation recovery failed")

    def send_action(self, action):
        target_rotvec = _rotvec_from_action(action)
        if self._enabled:
            self._log_pose("teleop/target_ee", self._target_pose(action))
        result = self._robot.send_action(self._user_to_env_action(action))
        self._check_orientation_watchdog(action, target_rotvec)
        self._log_diagnostics(action, target_rotvec)
        if self._enabled and self._last_observation is not None:
            try:
                self._log_pose("teleop/observed_ee", self._observed_pose(self._last_observation))
            except Exception:
                logger.exception("Failed to log observed EE pose")
        return result


def _add_workspace_paths() -> None:
    project_root = Path(__file__).resolve().parents[5]
    for rel in ("lerobot-sim/src", "so101-nexus/src"):
        path = project_root / rel
        if path.exists():
            sys.path.insert(0, str(path))


def _str_to_bool(value: str) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_optional_int(value: str) -> int | None:
    lowered = str(value).strip().lower()
    if lowered in {"none", "null", ""}:
        return None
    return int(value)


def _parse_rotation_degrees(value: str) -> int:
    text = str(value).strip().lower()
    if text.startswith("rotate"):
        text = text[len("rotate") :]
    elif text.startswith("rot"):
        text = text[len("rot") :]
    if text in {"0", "90", "180", "270"}:
        return int(text)
    raise argparse.ArgumentTypeError(
        "Rotation must be one of: 0, 90, 180, 270; or names like rotate90 / rotate180"
    )


class _TeeStream:
    def __init__(self, terminal_stream, log_stream):
        self._terminal_stream = terminal_stream
        self._log_stream = log_stream

    def write(self, text: str) -> int:
        written = self._terminal_stream.write(text)
        self._log_stream.write(text)
        return written

    def flush(self) -> None:
        self._terminal_stream.flush()
        self._log_stream.flush()


def _setup_run_log_tee() -> tuple[Path, object]:
    project_root = Path(__file__).resolve().parents[5]
    log_dir = project_root / "lerobot-sim/src/lerobot/teleoperators/handtracker/logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"handtracker_{timestamp}_{os.getpid()}.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    os.environ["HANDTRACKER_LOG_PATH"] = str(log_path)
    return log_path, log_handle


def _send_default_rerun_blueprint() -> None:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        from lerobot.utils import rerun_visualization
    except Exception as exc:
        logging.warning("Failed to import Rerun blueprint API: %s", exc)
        return

    blueprint = rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(origin="observation.overhead"),
            rrb.Spatial2DView(origin="observation.wrist"),
            rrb.Spatial2DView(origin="hand_tracker/side"),
            rrb.Spatial2DView(origin="hand_tracker/top"),
            rrb.TextDocumentView(origin="hand_tracker/control_status"),
            rrb.TextDocumentView(origin="hand_tracker/status"),
            rrb.TimeSeriesView(origin="hand_tracker/scale_valid_side"),
            rrb.TimeSeriesView(origin="hand_tracker/scale_valid_top"),
            rrb.TimeSeriesView(origin="hand_tracker/metric_source"),
            rrb.TimeSeriesView(origin="hand_tracker/dx_norm"),
            rrb.TimeSeriesView(origin="hand_tracker/dy_norm"),
            rrb.TimeSeriesView(origin="hand_tracker/xy_scale"),
            rrb.TimeSeriesView(origin="hand_tracker/dz_norm"),
            rrb.TimeSeriesView(origin="hand_tracker/dz_m"),
            rrb.TimeSeriesView(origin="hand_tracker/dx_m"),
            rrb.TimeSeriesView(origin="hand_tracker/dy_m"),
            rrb.TimeSeriesView(origin="hand_tracker/px_per_m_side"),
            rrb.TimeSeriesView(origin="hand_tracker/px_per_m_top"),
            rrb.TimeSeriesView(origin="hand_tracker/pitch_2d_deg"),
            rrb.TimeSeriesView(origin="hand_tracker/pitch_3d_deg"),
            rrb.TimeSeriesView(origin="hand_tracker/roll_deg"),
            rrb.TimeSeriesView(origin="hand_tracker/roll_valid"),
            rrb.TimeSeriesView(origin="hand_tracker/gripper_fraction"),
            rrb.TimeSeriesView(origin="teleop/target_ee"),
            rrb.TimeSeriesView(origin="teleop/observed_ee"),
            rrb.TimeSeriesView(origin="teleop/target_pitch_down_deg"),
            rrb.TimeSeriesView(origin="teleop/observed_pitch_down_deg"),
            rrb.TimeSeriesView(origin="teleop/orientation_error_deg"),
            rrb.TimeSeriesView(origin="teleop/position_error_mm"),
            rrb.TimeSeriesView(origin="action"),
        )
    )
    rerun_visualization.log_rerun_data.blueprint = blueprint
    rr.send_blueprint(blueprint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", default="my_awesome_follower_arm", help="Follower robot id")
    parser.add_argument("--env-id", default="MuJoCoTouch-v1", help="MuJoCo env id")
    parser.add_argument("--fps", type=int, default=30, help="Teleop loop FPS")
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array"),
        default="rgb_array",
        help="MuJoCo render mode; use 'human' for visual window",
    )
    parser.add_argument("--teleop-time-s", type=float, default=None, help="Optional session duration in seconds")
    parser.add_argument(
        "--display-data",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Show Rerun visualization; accepts true/false or use --no-display-data",
    )
    parser.add_argument("--no-display-data", dest="display_data", action="store_false", help="Disable Rerun display")
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--overhead-width", type=int, default=640)
    parser.add_argument("--overhead-height", type=int, default=480)
    parser.add_argument(
        "--handtracker-module",
        default="lerobot.teleoperators.handtracker.teleop_vision",
        help="Optional python module pathname providing get_gripper_fraction()",
    )
    parser.add_argument("--side-cam", type=_parse_optional_int, default=None, help="Side camera index or none")
    parser.add_argument("--top-cam", type=_parse_optional_int, default=None, help="Top camera index or none")
    parser.add_argument(
        "camera_options",
        nargs="*",
        default=[],
        help="Optional top camera transforms, in any order: rotate90/rotate180/rotate270, xmirror, ymirror. "
        "e.g. --top-cam 1 rotate180 xmirror. No mirroring is applied by default.",
    )
    parser.add_argument(
        "--wrist-roll-offset-deg",
        type=float,
        default=90.0,
        help="Roll applied to the start wrist orientation about the gripper axis (+ = clockwise looking along the gripper)",
    )
    parser.add_argument("--side-xmirror", action="store_true", help="Mirror side camera left/right")
    parser.add_argument("--side-ymirror", action="store_true", help="Mirror side camera up/down")
    parser.add_argument(
        "--side-rotation",
        type=_parse_rotation_degrees,
        default=0,
        help="Rotate side camera input before hand detection: 0, 90, 180, 270 or rotate90/rotate180",
    )
    parser.add_argument(
        "--top-rotation",
        type=_parse_rotation_degrees,
        default=0,
        help="Rotate top camera input before hand detection: 0, 90, 180, 270 or rotate90/rotate180",
    )
    args = parser.parse_args()

    args.top_xmirror = False
    args.top_ymirror = False
    for option in args.camera_options:
        name = option.strip().lower()
        if name == "xmirror":
            args.top_xmirror = True
        elif name == "ymirror":
            args.top_ymirror = True
        else:
            try:
                args.top_rotation = _parse_rotation_degrees(name)
            except argparse.ArgumentTypeError:
                parser.error(
                    f"Unknown camera option '{option}'. Expected rotate90/rotate180/rotate270, xmirror or ymirror."
                )

    return args


def main() -> None:
    args = parse_args()

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_path, log_handle = _setup_run_log_tee()
    sys.stdout = _TeeStream(original_stdout, log_handle)
    sys.stderr = _TeeStream(original_stderr, log_handle)

    _add_workspace_paths()

    import so101_nexus.lerobot_adapter  # type: ignore[import-not-found]  # noqa: F401 - register simulator plugin types
    from lerobot.robots import make_robot_from_config
    from lerobot.processor.factory import make_default_processors
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, init_logging, teleop_loop
    from lerobot.teleoperators.keyboard.so101_keyboard import KeyboardSOLeaderConfig
    from lerobot.teleoperators.handtracker import teleop_vision as handtracker_vision
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization
    from so101_nexus.teleop.session import build_sim_follower_config, prepare_follower_calibration  # type: ignore[import-not-found]
    calibration_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/sim_so_follower"
    prepare_follower_calibration(calibration_dir=calibration_dir, robot_id=args.robot_id)

    follower_config = build_sim_follower_config(
        env_id=args.env_id,
        robot_id=args.robot_id,
        wrist_wh=(args.wrist_width, args.wrist_height),
        overhead_wh=(args.overhead_width, args.overhead_height),
        fps=args.fps,
        calibration_dir=calibration_dir,
    )
    follower_config.env_kwargs["render_mode"] = args.render_mode
    follower_config.env_kwargs["control_mode"] = "pd_ee_pose"

    teleop_cfg = KeyboardSOLeaderConfig(
        handtracker_module=args.handtracker_module, wrist_roll_offset_deg=args.wrist_roll_offset_deg
    )
    cfg = TeleoperateConfig(
        teleop=teleop_cfg,
        robot=follower_config,
        fps=args.fps,
        teleop_time_s=args.teleop_time_s,
        display_data=args.display_data,
    )

    print(f"Starting keyboard teleop against sim SO101 follower... logging to {log_path}")
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(cfg.display_mode, session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
        if cfg.display_mode == "rerun":
            _send_default_rerun_blueprint()

    handtracker_vision.enable_rerun_logging(cfg.display_data and cfg.display_mode == "rerun", mode="off")
    handtracker_vision.init_default_service(
        args.side_cam,
        args.top_cam,
        args.side_rotation,
        args.top_rotation,
        (args.side_xmirror, args.side_ymirror),
        (args.top_xmirror, args.top_ymirror),
    )
    try:
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        try:
            from so101_nexus.kinematics import (  # type: ignore[import-not-found]
                EE_IK_ITERATIONS,
                EE_ORIENTATION_WEIGHT,
            )

            logging.info(
                "Sim IK config: EE_IK_ITERATIONS=%s, EE_ORIENTATION_WEIGHT=%s",
                EE_IK_ITERATIONS,
                EE_ORIENTATION_WEIGHT,
            )
        except Exception as exc:
            logging.warning("Could not read sim IK config: %s", exc)

        teleop = make_teleoperator_from_config(cfg.teleop)
        robot = _RerunEeLoggingRobot(
            make_robot_from_config(cfg.robot),
            enabled=cfg.display_data and cfg.display_mode == "rerun",
        )
        robot.set_teleop(teleop)
        teleop.connect()
        robot.connect()

        initial_observation = robot.get_observation()
        initial_pose = robot.get_observed_ee_pose(initial_observation)
        if initial_pose:
            logging.info("Initial observed EE pose: %s", initial_pose)
            gripper_value = float(initial_observation.get("gripper.pos", 0.0))
            initial_pose = _warmup_to_seed_position(robot, initial_pose, gripper_value)
            logging.info("Seeding teleop from warmed-up EE pose: %s", initial_pose)
        if hasattr(teleop, "seed_initial_pose") and initial_pose:
            teleop.seed_initial_pose(initial_pose)
        robot.enable_watchdog()

        try:
            teleop_loop(
                teleop=teleop,
                robot=robot,
                fps=cfg.fps,
                display_data=cfg.display_data,
                display_mode=cfg.display_mode,
                duration=cfg.teleop_time_s,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                display_compressed_images=(
                    True
                    if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
                    else cfg.display_compressed_images
                ),
            )
        finally:
            if cfg.display_data:
                shutdown_visualization(cfg.display_mode)
            teleop.disconnect()
            robot.disconnect()
    finally:
        handtracker_vision.enable_rerun_logging(False)
        handtracker_vision.shutdown_default_service()
        sys.stdout.flush()
        sys.stderr.flush()
        log_handle.flush()
        log_handle.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    main()
