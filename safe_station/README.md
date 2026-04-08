---
title: Safe Station Environment Server
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - ev-management
---

# Safe Station Environment ⚡

An advanced Reinforcement Learning environment for managing a hard-constrained EV Smart-Station. The agent's goal is to maximize profit and customer turnover using a Dual Source power system (Grid + Station Battery BESS) while respecting dynamic peak pricing algorithms.

---

## 🌎 Environment Overview 

The Smart-Station handles dynamically arriving EVs that demand a certain amount of battery charge. Electricity must be smartly managed across different hours of the day where `grid_price` heavily fluctuates:
- **Off-Peak (00:00 - 06:00)**: ₹4.0 - Optimal for topping up the station battery.
- **Normal (07:00 - 17:00, 23:00)**: ₹8.0
- **Peak Hour (18:00 - 22:00)**: ₹15.0 - Highly expensive. Drawing from the grid should be heavily avoided using peak-shaving tactics.

## 📊 State Space (Observation)
The environment observation space (`SafeStationObservation`) gives the agent real-time metrics to act on:
- `hour` (int): Hour of the day (0-23).
- `grid_price` (float): Current cost to pull electricity from the grid.
- `station_battery_level` (float): The current level of the Station's BESS (0.0 to 100.0).
- `car_present` (int): 0 if the station is empty, 1 if an EV is waiting.
- `car_battery_need` (float): Amount of electricity the present EV still needs (0.0 to 100.0).

## 🎮 Action Space
`SafeStationAction` defines the physical actions the RL Controller can execute on each Step (0 to 3):
- **`0: Grid Charge`**: Charge the EV by drawing exactly 20 units strictly from the Grid.
- **`1: Battery Charge`**: Charge the EV using 20 units cleanly from the internal Station Battery (Peak Shave).
- **`2: Top-Up Station Battery`**: If the station is empty, draw up to 30 units from the Grid to refill the offline BESS.
- **`3: Hybrid Charge`**: Rapid fast-charge an EV with 40 units (20 from Grid, 20 from Station Battery).

## 🏆 Rewards Breakdown
The RL agent logic is strictly optimized by balancing these backend Mathematical Rewards:

### ✅ Positive Rewards (+)
* **`+20.0` (Strategic Top-Up Bonus)**: Refilling the battery during ultra-cheap off-peak hours (Grid `< ₹5.0`).
* **`+50.0` (Peak-Shave Bonus)**: Expending power from the station's internal battery during punishing Peak Hours (Grid `> ₹10.0`).
* **`+200.0` (Success Bonus)**: Successfully clearing an EV's battery need to `0`, allowing the car to cleanly leave.

### ❌ Negative Rewards (-)
* **Operational Cost**: Any power drawn from the Grid incurs a penalty proportionate to the time of day: `-(Grid Units Used × Grid Price)`.
* **`-5.0` (Time Penalty)**: Continuous penalty for every step an EV is kept waiting in the station. Reduces overall throughput.
* **`-500.0` (Critical Failure)**: Lethal penalty terminating the episode if the station carelessly drops its BESS level to exactly `0.0`.

---

## ⚙️ Core API Interfaces

### `reset()`
Resets the environment back to a clean starting state.
- Initializes a new randomized `hour` and calculates the corresponding `grid_price`.
- Restores the `station_battery_level` safely back to `100.0`.
- Enacts a 50% chance of spawning an initial EV (`car_present = 1`) with a random `car_battery_need`.
- **Returns**: A `SafeStationObservation` initial state object (In local Python wrappers, translated into a Dictionary).

### `step(action)`
Executes the provided action (0-3) and mathematically advances the environment simulation.
- Validates grid transactions and securely deducts the Operational Cost for any Grid power used.
- Assesses EV charging logic and dynamically generates combined Positive/Negative Rewards.
- Advances the internal simulation clock (Every 4 steps = 1 physical Hour).
- Has a 25% chance of spawning a new EV every 15-minute frame if the station is completely empty.
- **Returns**: Exposes `(observation, reward, done)` containing the updated metrics.

### `state()` / Properties
A static tracking function handling the backend simulation limits:
- Tracks the unique `episode_id`.
- Tally of the current `step_count` to ensure the episode cleanly truncates at exactly the 96-steps limit (24 hours standard).

---

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv logic stack to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The deployed space will automatically include:
- **Web Interface** at `/web`
- **API Documentation** at `/docs`
- **WebSocket** at `/ws`
