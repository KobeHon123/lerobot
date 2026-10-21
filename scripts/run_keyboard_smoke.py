#!/usr/bin/env python3
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root.parent / 'so101-nexus' / 'src'))
import so101_nexus.lerobot_adapter  # register sim plugin
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators.keyboard.so101_keyboard import KeyboardSOLeaderConfig
from so101_nexus.teleop.session import build_sim_follower_config, prepare_follower_calibration

calibration_dir = Path.home() / '.cache/huggingface/lerobot/calibration/robots/sim_so_follower'
prepare_follower_calibration(calibration_dir=calibration_dir, robot_id='my_awesome_follower_arm')

follower_config = build_sim_follower_config(
    env_id='MuJoCoTouch-v1',
    robot_id='my_awesome_follower_arm',
    wrist_wh=(640,480),
    overhead_wh=(640,480),
    fps=30,
    calibration_dir=calibration_dir,
)

follower_config.env_kwargs['render_mode'] = 'rgb_array'
teleop_cfg = KeyboardSOLeaderConfig()
cfg = TeleoperateConfig(
    teleop=teleop_cfg,
    robot=follower_config,
    fps=30,
    teleop_time_s=5.0,
    display_data=False,
)
print('Starting teleoperate smoke test (5s)...')
teleoperate(cfg)
print('Teleoperate finished')
