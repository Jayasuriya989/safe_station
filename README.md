---
title: Safe Station Environment Server
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - ev-management
---

# Safe Station Environment ⚡

An advanced Reinforcement Learning environment for managing a hard-constrained EV Smart-Station. This Space runs the environment server using the **OpenEnv** framework.

## 🚀 Quick Start (Local)

If you want to run this environment locally using Docker:

```bash
# Build the image
docker build -t safe-station .

# Run the container
docker run -p 8000:8000 safe-station
```

## 📊 Environment Specifications

- **Goal**: Maximize profit by smart charging EVs while minimizing grid costs.
- **State Space**: `hour`, `grid_price`, `station_battery_level`, `car_present`, `car_battery_need`.
- **Action Space**:
  - `0`: Grid Charge (20 units from Grid)
  - `1`: Battery Charge (20 units from BESS)
  - `2`: Top-Up BESS (Refill up to 30 units)
  - `3`: Hybrid Charge (40 units total - 20/20 split)

## 🏆 Reward System

- **+200.0**: Successfully charging a car to completion.
- **+50.0**: Peak-Shaving (using battery during expensive hours).
- **+Profit**: Reward for topping up during off-peak hours (₹4.0).
- **-Cost**: Penalties for grid usage and car wait times.

---
*Developed for the Meta PyTorch OpenEnv Hackathon 2026.*
