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
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from pprint import pformat

import numpy as np


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

        if self._enabled:
            from lerobot.model import RobotKinematics
            from lerobot.robots.so_follower.robot_kinematic_processor import compute_forward_kinematics_joints_to_ee
            from so101_nexus import get_so101_urdf_path  # type: ignore[import-not-found]
            from so101_nexus.config import SO101_TCP_FRAME_NAME  # type: ignore[import-not-found]

            self._compute_forward_kinematics_joints_to_ee = compute_forward_kinematics_joints_to_ee
            self._kinematics = RobotKinematics(str(get_so101_urdf_path()), SO101_TCP_FRAME_NAME)
            self._motor_names = list(getattr(robot, "motors", []))

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
        cv2.putText(canvas, "origin", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, origin_color, 1)
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
        self._overlay_overhead_grid(observation)
        self._last_observation = observation
        return observation

    def send_action(self, action):
        if self._enabled:
            self._log_pose("teleop/target_ee", self._target_pose(action))
        result = self._robot.send_action(self._user_to_env_action(action))
        if self._enabled and self._last_observation is not None:
            try:
                self._log_pose("teleop/observed_ee", self._observed_pose(self._last_observation))
            except Exception:
                logger = logging.getLogger(__name__)
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
            rrb.TextDocumentView(origin="hand_tracker/status"),
            rrb.TimeSeriesView(origin="hand_tracker/dx_norm"),
            rrb.TimeSeriesView(origin="hand_tracker/dy_norm"),
            rrb.TimeSeriesView(origin="hand_tracker/xy_scale"),
            rrb.TimeSeriesView(origin="hand_tracker/dz_norm"),
            rrb.TimeSeriesView(origin="hand_tracker/pitch_2d_deg"),
            rrb.TimeSeriesView(origin="hand_tracker/pitch_3d_deg"),
            rrb.TimeSeriesView(origin="hand_tracker/gripper_fraction"),
            rrb.TimeSeriesView(origin="teleop/target_ee"),
            rrb.TimeSeriesView(origin="teleop/observed_ee"),
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
        teleop = make_teleoperator_from_config(cfg.teleop)
        robot = _RerunEeLoggingRobot(
            make_robot_from_config(cfg.robot),
            enabled=cfg.display_data and cfg.display_mode == "rerun",
        )
        teleop.connect()
        robot.connect()

        initial_observation = robot.get_observation()
        initial_pose = robot.get_observed_ee_pose(initial_observation)
        if hasattr(teleop, "seed_initial_pose") and initial_pose:
            logging.info("Initial observed EE pose for seeding: %s", initial_pose)
            teleop.seed_initial_pose(initial_pose)

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
