# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
import os
import re
from typing import Any

import numpy as np
import rerun as rr

from .constants import OBS_PREFIX, OBS_STR


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format and logged as `rr.Image`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"
            is_wowskin = "wowskin" in key
            is_motor_state = key.endswith(".pos")
            motor_obs_key = f"motor/observation/{k}"
            wowskin_scalar_match = re.match(r"^observation\.wowskin_(\d+)_(x|y|z)$", key)

            if _is_scalar(v):
                if is_motor_state:
                    rr.log(motor_obs_key, rr.Scalars(float(v)))
                elif wowskin_scalar_match:
                    mag_idx, axis_label = wowskin_scalar_match.groups()
                    rr.log(f"wowskin/mag_{mag_idx}/{axis_label}", rr.Scalars(float(v)))
                else:
                    rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    if is_wowskin and arr.size % 3 == 0:
                        mag_count = arr.size // 3
                        axis_labels = ("x", "y", "z")
                        for mag_idx in range(mag_count):
                            base = mag_idx * 3
                            for axis_idx, axis_label in enumerate(axis_labels):
                                rr.log(
                                    f"wowskin/mag_{mag_idx}/{axis_label}",
                                    rr.Scalars(float(arr[base + axis_idx])),
                                )
                    elif is_wowskin:
                        for i, vi in enumerate(arr):
                            rr.log(f"wowskin/{i}", rr.Scalars(float(vi)))
                    else:
                        for i, vi in enumerate(arr):
                            rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    rr.log(key, rr.Image(arr), static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"
            motor_action_key = f"motor/action/{k}"

            if _is_scalar(v):
                rr.log(motor_action_key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{motor_action_key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{motor_action_key}_{i}", rr.Scalars(float(vi)))
