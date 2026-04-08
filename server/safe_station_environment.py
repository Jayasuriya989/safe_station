# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Safe Station Environment Implementation.
An RL environment for managing a hard-constrained EV Smart-Station.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SafeStationAction, SafeStationObservation
except ImportError:
    from models import SafeStationAction, SafeStationObservation


class SafeStationEnvironment(Environment):
    """
    Safe Station Environment.

    Maximizes profit and customer turnover using a Dual Source power system.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the safe_station environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        
        # Internal state
        self.hour = 0
        self.grid_price = 4.0
        self.station_battery_level = 100.0
        self.car_present = 0
        self.car_battery_need = 0.0

    def _get_grid_price(self, hour: int) -> float:
        """Returns the grid price for a given hour."""
        if 18 <= hour <= 22:
            return 15.0  # Peak price
        elif 0 <= hour <= 6:
            return 4.0   # Off-peak price
        else:
            return 8.0   # Normal price

    def _get_observation(self, reward: float = 0.0, done: bool = False) -> SafeStationObservation:
        return SafeStationObservation(
            hour=self.hour,
            grid_price=self.grid_price,
            station_battery_level=self.station_battery_level,
            car_present=self.car_present,
            car_battery_need=self.car_battery_need,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count}
        )

    def reset(self) -> SafeStationObservation:
        """Reset the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        self.hour = random.randint(0, 23)
        self.grid_price = self._get_grid_price(self.hour)
        self.station_battery_level = 100.0
        
        # Start with a car 50% of the time
        self.car_present = 1 if random.random() > 0.5 else 0
        self.car_battery_need = float(random.randint(20, 80)) if self.car_present else 0.0

        return self._get_observation()

    def step(self, action: SafeStationAction) -> SafeStationObservation:  # type: ignore[override]
        """Execute action in the EV station."""
        self._state.step_count += 1
        act = action.action
        
        reward = 0.0
        done = False
        
        # Ensure agent is physically restricted from choosing 'Wait' (Action 2) if car is present
        if self.car_present == 1 and act == 2:
            # Enforce hard constraint by defaulting to Grid Charge
            act = 0
            
        energy_from_grid = 0.0
        energy_from_battery = 0.0
        car_charge_amt = 0.0
        
        if act == 0:  # Grid Charge
            if self.car_present:
                energy_from_grid = 20.0
                car_charge_amt = 20.0
        elif act == 1:  # Battery Charge
            if self.car_present:
                energy_from_battery = 20.0
                car_charge_amt = 20.0
        elif act == 2:  # Top-up Station Battery
            if not self.car_present:
                # Top up BESS
                space_left = 100.0 - self.station_battery_level
                energy_from_grid = min(30.0, space_left)
                self.station_battery_level += energy_from_grid
                
                # Strategic Top-Up Bonus (Rewards storing cheap energy)
                if self.grid_price < 5.0 and energy_from_grid > 0:
                    reward += (energy_from_grid * 6.0)
        elif act == 3:  # Hybrid Charge
            if self.car_present:
                # fast charge: 20 from grid, 20 from BESS
                energy_from_grid = 20.0
                energy_from_battery = 20.0
                car_charge_amt = 40.0
                
        # Drain battery
        self.station_battery_level -= energy_from_battery
        
        # Apply Operational Cost (-Price for grid usage)
        reward -= (energy_from_grid * self.grid_price)

        
        # Process Car charging
        if self.car_present and car_charge_amt > 0:
            actual_charge = min(car_charge_amt, self.car_battery_need)
            self.car_battery_need -= actual_charge
            
            # Peak-Shave Bonus
            if energy_from_battery > 0 and self.grid_price > 10.0:
                reward += 50.0
                
            # Success Bonus
            if self.car_battery_need <= 0:
                self.car_battery_need = 0.0
                reward += 200.0
                self.car_present = 0 # Car leaves
                
        # Time Penalty / Inefficiency
        if self.car_present:
            reward -= 5.0
            
        # Critical Failure
        if self.station_battery_level <= 0:
            self.station_battery_level = 0.0
            reward -= 500.0
            done = True
            
        # Time progression
        # Let's say each step is 15 mins. 4 steps = 1 hour.
        if self._state.step_count % 4 == 0:
            self.hour = (self.hour + 1) % 24
            self.grid_price = self._get_grid_price(self.hour)
            
        # Random car arrival if no car
        if not self.car_present and not done:
            if random.random() < 0.25: # 25% chance per 15 min step
                self.car_present = 1
                self.car_battery_need = float(random.randint(20, 80))
                
        # Episode length limit (e.g. 1 day = 96 steps)
        if self._state.step_count >= 96:
            done = True

        return self._get_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state
