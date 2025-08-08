# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 MiroMind Team
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
"""
A unified tracking interface that supports logging data to different backend
"""

from typing import List, Union

from verl.utils.tracking import Tracking

from wandb import AlertLevel


# Adapted from verl/utils/tracking.py
# Updated by MiroMind Team
# 1. Add alert function
class MonitorTracking(Tracking):
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: Union[str, List[str]] = "console",
        config=None,
    ):
        super().__init__(project_name, experiment_name, default_backend, config)

    def alert(self, title, text, level=AlertLevel.WARN, wait_duration=180):
        assert "wandb" in self.logger, "wandb is not in the tracking"
        self.logger["wandb"].alert(
            title=title,
            text=text,
            level=level,
            wait_duration=wait_duration,
        )
