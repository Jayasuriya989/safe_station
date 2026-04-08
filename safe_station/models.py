# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Safe Station Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SafeStationAction(Action):
    """Action for the Safe Station environment."""

    action: int = Field(
        ..., 
        description="0: Grid Charge, 1: Battery Charge, 2: Top-up Station Battery, 3: Hybrid Charge",
        ge=0,
        le=3
    )


class SafeStationObservation(Observation):
    """Observation from the Safe Station environment."""

    hour: int = Field(default=0, description="Hour of the day (0-23)", ge=0, le=23)
    grid_price: float = Field(default=0.0, description="Current price of the grid")
    station_battery_level: float = Field(default=100.0, description="Station Battery Level (0-100)", ge=0.0, le=100.0)
    car_present: int = Field(default=0, description="Car Presence (0 or 1)", ge=0, le=1)
    car_battery_need: float = Field(default=0.0, description="Car Battery Need (0-100)", ge=0.0, le=100.0)
