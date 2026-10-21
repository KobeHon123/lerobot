#!/usr/bin/env python3

"""Launch LeRobot teleoperation against the SO101-Nexus simulator follower.

This is the lightweight, repo-local equivalent of the old wrapper currently kept
under ``so101-nexus/scripts``. It adds the local ``lerobot-sim`` and sibling
``so101-nexus`` source trees to ``sys.path``, registers the simulator plugin,
creates the synthetic follow-up calibration file, wires both MuJoCo camera
observations into the simulator env, and then calls the official LeRobot
teleoperate API with Rerun enabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_workspace_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sibling_repo = repo_root.parent / "so101-nexus"
    for path in (repo_root / "src", sibling_repo / "src"):
        if path.exists():
            sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", required=True, help="Serial port for the physical SO101 leader")
    parser.add_argument(
        "--leader-id",
        default="my_awesome_leader_arm",
        help="LeRobot leader id used for calibration lookup",
    )
    parser.add_argument(
        "--robot-id",
        default="my_awesome_follower_arm",
        help="LeRobot follower id used for calibration lookup",
    )
    parser.add_argument("--env-id", default="MuJoCoTouch-v1", help="Registered MuJoCo env id to launch")
    parser.add_argument(
        "--calibration-dir",
        default=str(Path.home() / ".cache/huggingface/lerobot/calibration/robots/sim_so_follower"),
        help="Directory holding the synthetic follower calibration file",
    )
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--overhead-width", type=int, default=640)
    parser.add_argument("--overhead-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="Camera fps and teleop loop target")
    parser.add_argument(
        "--render-mode",
        choices=("rgb_array", "human"),
        default="rgb_array",
        help="MuJoCo render mode to pass into the simulator env",
    )
    parser.add_argument(
        "--teleop-time-s",
        type=float,
        default=None,
        help="Optional duration limit for the session",
    )
    parser.add_argument(
        "--display-data",
        dest="display_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the Rerun telemetry window",
    )
    return parser.parse_args()


def main() -> None:
    _add_workspace_paths()

    import so101_nexus.lerobot_adapter  # noqa: F401 - register simulator plugin types
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
    from lerobot.teleoperators.so_leader import SO101LeaderConfig
    from so101_nexus.teleop.session import build_sim_follower_config, prepare_follower_calibration

    args = parse_args()
    calibration_dir = Path(args.calibration_dir).expanduser()
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

    teleop_config = SO101LeaderConfig(port=args.leader_port, id=args.leader_id)
    cfg = TeleoperateConfig(
        teleop=teleop_config,
        robot=follower_config,
        fps=args.fps,
        teleop_time_s=args.teleop_time_s,
        display_data=args.display_data,
    )
    teleoperate(cfg)


if __name__ == "__main__":
    main()
