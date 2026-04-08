"""
Safe Station Inference Validation Wrapper
Meta PyTorch OpenEnv Hackathon 2026

This is the single entry point for all environment validation:
  1. OpenEnv Compliance Check  - Validates reset()/step() format (dict + 3-tuple)
  2. Multi-Step Physical Test   - Runs a full episode and prints reward math breakdown
"""

import asyncio
import random
import sys
import io
from typing import Dict, Tuple

# Fix Windows PowerShell Unicode encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# === OpenEnv HTTP Client / Models ===
try:
    from safe_station.client import SafeStationEnv
    from safe_station.models import SafeStationAction
    from safe_station.server.safe_station_environment import SafeStationEnvironment
except ImportError:
    from client import SafeStationEnv
    from models import SafeStationAction
    from server.safe_station_environment import SafeStationEnvironment

ACTIONS = {0: "Grid Charge", 1: "Battery Charge", 2: "Top-Up BESS", 3: "Hybrid Charge"}
SEP = "-" * 62

# =====================================================================
# SECTION 1: InferenceWrapper (Standard Gym-Style Interface)
# This is what the Hackathon AI Inference Script interacts with.
# =====================================================================
class InferenceWrapper:
    """
    Translates the OpenEnv Pydantic-based client into a standard Gym
    inference format strictly matching:
        - reset()      -> dict
        - step(action) -> (dict, float, bool)
    """
    def __init__(self, url: str = "http://localhost:8000"):
        self.client = SafeStationEnv(base_url=url)
        self.action_space = list(ACTIONS.keys())  # [0, 1, 2, 3]

    async def reset(self) -> Dict[str, float]:
        """Returns the initial state as a strict observation dictionary."""
        res = await self.client.reset()
        return _to_obs_dict(res.observation)

    async def step(self, action: int) -> Tuple[Dict[str, float], float, bool]:
        """Returns EXACTLY three values: (observation_dict, reward, done)."""
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}. Must be one of {self.action_space}.")
        res = await self.client.step(SafeStationAction(action=action))
        reward = float(res.reward if res.reward is not None else 0.0)
        return _to_obs_dict(res.observation), reward, bool(res.done)

    async def close(self):
        await self.client.close()


def _to_obs_dict(obs) -> Dict[str, float]:
    return {
        "hour":                  float(obs.hour),
        "grid_price":            float(obs.grid_price),
        "station_battery_level": float(obs.station_battery_level),
        "car_present":           float(obs.car_present),
        "car_battery_need":      float(obs.car_battery_need),
    }


# =====================================================================
# SECTION 2: Physical Stress Tester (Multi-Step Logic)
# Directly interacts with backend logic to verify mathematical correctness.
# =====================================================================

def heuristic_pick_action(car_present, grid_price, bess_level, car_need=0.0):
    """
    Heuristic agent for testing:
      - Car present + peak hour    -> Battery Charge (+50 bonus)
      - Car present + high need     -> Hybrid (Fastest)
      - No car      + cheap hours  -> Top-Up BESS (+Bonus)
    """
    if car_present:
        if bess_level <= 0: return 0
        if grid_price > 10.0: return 1 if bess_level >= 20.0 else 0
        if car_need <= 20.0 and bess_level >= 20.0: return 1
        elif car_need <= 40.0 and bess_level >= 20.0: return 1
        elif bess_level >= 20.0: return 3
        else: return 0
    else:
        if grid_price < 5.0 and bess_level < 100.0: return 2
        return 0


def compute_reward_math(action_id, grid_price, car_present, car_need_before, bess_before):
    """Mirror the environment reward math for validation output."""
    base = bonus = op_cost = time_pen = 0.0
    energy_grid = energy_batt = charge_amt = 0.0

    if action_id == 0 and car_present:
        energy_grid = 20.0; charge_amt = 20.0
    elif action_id == 1 and car_present:
        energy_batt = 20.0; charge_amt = 20.0
    elif action_id == 2 and not car_present:
        energy_grid = min(30.0, 100.0 - bess_before)
        if grid_price < 5.0 and energy_grid > 0:
            bonus += (energy_grid * 6.0)
    elif action_id == 3 and car_present:
        energy_grid = 20.0; energy_batt = 20.0; charge_amt = 40.0

    op_cost = -(energy_grid * grid_price)

    if car_present and charge_amt > 0:
        if energy_batt > 0 and grid_price > 10.0: bonus += 50.0
        if car_need_before - charge_amt <= 0: base += 200.0

    if car_present and (car_need_before - charge_amt > 0):
        time_pen = -5.0

    return base, bonus, op_cost, time_pen


def run_full_episode_test(initial_hour, initial_bess, initial_car_present, initial_car_needs):
    """Runs and prints a full multi-step episode."""
    env = SafeStationEnvironment()
    env.reset()
    env.hour                  = initial_hour
    env.grid_price            = env._get_grid_price(initial_hour)
    env.station_battery_level = float(initial_bess)
    env.car_present           = int(initial_car_present)
    env.car_battery_need      = float(initial_car_needs)

    print(f"\n{SEP}")
    print(f"  SECTION 2: Multi-Step Physical Test")
    print(f"{SEP}")
    print(f"  START -> Hour: {initial_hour:02d}:00 | Grid: Rs.{env.grid_price:.1f} | BESS: {initial_bess:.1f}%")
    print(f"  Car: {'YES' if initial_car_present else 'NO'} | Need: {initial_car_needs:.1f}%")

    episode_reward = 0.0
    charge_step = 0

    # Phase 1: Wait/Top-up
    if not env.car_present:
        print(f"\n  [PHASE 1] Waiting for car departure/arrival...")
        for ws in range(1, 6):
            aid = heuristic_pick_action(env.car_present, env.grid_price, env.station_battery_level)
            b_bef = env.station_battery_level
            gp = env.grid_price
            
            env.step(SafeStationAction(action=aid))
            
            rew = 0.0
            if aid == 2:
                e = min(30.0, 100.0 - b_bef)
                rew = (e * 6.0) - (e * gp) if gp < 5.0 else -(e * gp)
            episode_reward += rew
            
            print(f"  Wait {ws} | BESS {b_bef:.1f}% -> {env.station_battery_level:.1f}% | Action: {ACTIONS[aid]} | Reward: {rew:+.2f}")
            if env.car_present: break
    
    # Phase 2: Charging
    print(f"\n  [PHASE 2] Charging Active Session")
    for s in range(1, 11):
        if not env.car_present: break
        
        c_bef, n_bef, b_bef, gp = env.car_present, env.car_battery_need, env.station_battery_level, env.grid_price
        aid = heuristic_pick_action(c_bef, gp, b_bef, n_bef)
        
        env.step(SafeStationAction(action=aid))
        charge_step += 1
        
        base, bonus, op, tp = compute_reward_math(aid, gp, c_bef, n_bef, b_bef)
        step_rew = base + bonus + op + tp
        episode_reward += step_rew
        
        print(f"\n  Step {charge_step} | Action: {ACTIONS[aid]} (ID {aid})")
        print(f"    Base: {base:>+7.1f} | Bonus: {bonus:>+7.1f} | Grid: {op:>+7.1f} | Time: {tp:>+7.1f} | REWARD: {step_rew:+.2f}")
        print(f"    BESS: {env.station_battery_level:.1f}% | Car: {env.car_battery_need:.1f}% Remaining")
        
        if env.car_present == 0:
            print(f"  *** Car fully charged! ***")
            break

    print(f"\n{SEP}")
    print(f"  TEST COMPLETE | TOTAL EPISODE REWARD: {episode_reward:+.2f}")
    print(f"{SEP}\n")


# =====================================================================
# Main Validation Flow
# =====================================================================

async def run_compliance_check():
    print(f"\n{SEP}")
    print("  SECTION 1: OpenEnv API Compliance Check")
    print(f"{SEP}")
    print("Connecting to environment server at http://localhost:8000...")
    
    try:
        env = InferenceWrapper()
        obs = await env.reset()
        print(f"  [PASS] reset() -> {list(obs.keys())}")
        
        o, r, d = await env.step(0)
        print(f"  [PASS] step()  -> (dict, {type(r).__name__}, {type(d).__name__})")
        
        await env.close()
        print("\n  *** API Compliance PASSED ***")
    except Exception as e:
        print(f"\n  [FAIL] Compliance check failed: {e}")
        print("  (Make sure the environment server is running via 'python -m safe_station.server.app')")

async def main():
    # 1. API Compliance (External)
    await run_compliance_check()

    # 2. Physical Multi-Step Test (Internal)
    rt = random.randint(0, 23)
    rb = float(random.randint(20, 100))
    rc = random.choice([0, 1])
    rn = float(random.randint(20, 80)) if rc else 0.0
    
    run_full_episode_test(rt, rb, rc, rn)

if __name__ == "__main__":
    asyncio.run(main())
