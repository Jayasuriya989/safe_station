# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Safe Station Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SafeStationAction, SafeStationObservation
except ImportError:
    from models import SafeStationAction, SafeStationObservation


class SafeStationEnv(
    EnvClient[SafeStationAction, SafeStationObservation, State]
):
    """
    Client for the Safe Station Environment.
    """

    def _step_payload(self, action: SafeStationAction) -> Dict:
        """
        Convert SafeStationAction to JSON payload for step message.
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SafeStationObservation]:
        """
        Parse server response into StepResult[SafeStationObservation].
        """
        obs_data = payload.get("observation", {})
        observation = SafeStationObservation(
            hour=obs_data.get("hour", 0),
            grid_price=obs_data.get("grid_price", 0.0),
            station_battery_level=obs_data.get("station_battery_level", 0.0),
            car_present=obs_data.get("car_present", 0),
            car_battery_need=obs_data.get("car_battery_need", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
