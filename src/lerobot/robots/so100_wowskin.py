#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from .config import RobotConfig
from .so100_follower import SO100Follower, SO100FollowerConfig

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("so100_wowskin")
@dataclass
class SO100WowSkinConfig(SO100FollowerConfig):
    wowskin_port: str = "/dev/ttyACM2"
    wowskin_num_mags: int = 5
    wowskin_baseline_samples: int = 5
    wowskin_startup_s: float = 1.0


class SO100WowSkin(SO100Follower):
    name = "so100_wowskin"

    def __init__(self, config: SO100WowSkinConfig):
        super().__init__(config)
        self.config = config
        self._wowskin = None
        self._wowskin_baseline = None

    @property
    def is_connected(self) -> bool:
        wowskin_connected = self._wowskin is not None
        return super().is_connected and wowskin_connected

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Get parent motor state features
        parent_features = super().observation_features
        # Add WowSkin as a separate feature
        wowskin_features = {f"observation.wowskin": (self.config.wowskin_num_mags * 3,)}
        return {**parent_features, **wowskin_features}

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        self._init_wowskin()
        logger.info("%s WowSkin connected.", self)

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        # Add WowSkin as a separate vector (not individual named fields)
        wowskin_values = self._read_wowskin()
        obs["observation.wowskin"] = wowskin_values.astype(np.float32)
        return obs

    def disconnect(self) -> None:
        try:
            super().disconnect()
        finally:
            self._shutdown_wowskin()

    def reset_wowskin_baseline(self) -> None:
        if self._wowskin is None:
            raise RuntimeError("WowSkin is not initialized.")

        self._wowskin_baseline = self._compute_wowskin_baseline()
        logger.info("%s WowSkin baseline reset.", self)

    def _init_wowskin(self) -> None:
        from anyskin import AnySkinProcess

        if self._wowskin is not None:
            return

        self._wowskin = AnySkinProcess(num_mags=self.config.wowskin_num_mags, port=self.config.wowskin_port)
        self._wowskin.start()
        time.sleep(self.config.wowskin_startup_s)
        self._wowskin_baseline = self._compute_wowskin_baseline()

    def _compute_wowskin_baseline(self) -> np.ndarray:
        if self._wowskin is None:
            raise RuntimeError("WowSkin is not initialized.")

        baseline_data = self._wowskin.get_data(num_samples=self.config.wowskin_baseline_samples)
        baseline_array = np.asarray(baseline_data, dtype=np.float32)

        if baseline_array.size == 0:
            raise RuntimeError("WowSkin returned no data. Ensure the sensor is streaming.")

        if baseline_array.ndim == 1:
            baseline_array = np.atleast_2d(baseline_array)

        expected_dim = self.config.wowskin_num_mags * 3
        if baseline_array.shape[1] == expected_dim + 1:
            baseline_array = baseline_array[:, 1:]
        elif baseline_array.shape[1] != expected_dim:
            raise RuntimeError(
                f"Unexpected WowSkin sample size {baseline_array.shape[1]} (expected {expected_dim} or {expected_dim + 1})."
            )

        return baseline_array.mean(axis=0)

    def _read_wowskin(self) -> np.ndarray:
        if self._wowskin is None or self._wowskin_baseline is None:
            raise RuntimeError("WowSkin is not initialized.")

        sensor_samples = self._wowskin.get_data(num_samples=1)
        if not sensor_samples:
            raise RuntimeError("WowSkin returned no data. Ensure the sensor is streaming.")

        sensor_data = np.asarray(sensor_samples[0], dtype=np.float32)
        expected_dim = self.config.wowskin_num_mags * 3
        if sensor_data.ndim == 1 and sensor_data.shape[0] == expected_dim + 1:
            sensor_data = sensor_data[1:]
        elif sensor_data.ndim == 1 and sensor_data.shape[0] != expected_dim:
            raise RuntimeError(
                f"Unexpected WowSkin sample size {sensor_data.shape[0]} (expected {expected_dim} or {expected_dim + 1})."
            )

        data = np.asarray(sensor_data, dtype=np.float32)
        return data - self._wowskin_baseline

    def _shutdown_wowskin(self) -> None:
        if self._wowskin is None:
            return

        self._wowskin.pause_streaming()
        self._wowskin.join()
        self._wowskin = None
        self._wowskin_baseline = None
