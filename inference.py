"""
Safe Station Inference Validation Wrapper
Meta PyTorch OpenEnv Hackathon 2026

This is the single entry point for all environment validation.
"""

import asyncio
import random
import sys
import io
import os
import json
from typing import Dict, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Robust path injection: Ensure the repository root is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
server_dir = os.path.join(root_dir, "server")
if os.path.exists(server_dir) and server_dir not in sys.path:
    sys.path.insert(0, server_dir)

# === OpenEnv HTTP Client / Models ===
try:
    from client import SafeStationEnv
    from models import SafeStationAction
    from safe_station_environment import SafeStationEnvironment
except ImportError:
    try:
        from .client import SafeStationEnv
        from .models import SafeStationAction
        from .safe_station_environment import SafeStationEnvironment
    except ImportError:
        import client
        import models
        import safe_station_environment
        SafeStationEnv = client.SafeStationEnv
        SafeStationAction = models.SafeStationAction
        SafeStationEnvironment = safe_station_environment.SafeStationEnvironment

ACTIONS = {0: "Grid Charge", 1: "Battery Charge", 2: "Top-Up BESS", 3: "Hybrid Charge"}
SEP = "-" * 62

# =====================================================================
# SECTION 0: Global Configuration & Logic
# =====================================================================

def get_mandatory_vars() -> Dict[str, str]:
    """Returns the mandatory environment variables required for the Hackathon."""
    # 1. API_BASE_URL (Strictly use platform injected variables)
    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    
    # 2. API_KEY (Strictly use platform injected variables)
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # 3. Model Name
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    # Diagnostics (stderr is visible in dashboard logs but won't break stdout parsing)
    sys.stderr.write(f"\n[DIAGNOSTIC] Environment Init:\n")
    if not base_url:
        sys.stderr.write("  [WARNING] API_BASE_URL is missing! Local testing may fail.\n")
    if not api_key:
        sys.stderr.write("  [WARNING] API_KEY is missing! Using 'EMPTY' for initialization.\n")
    
    sys.stderr.write(f"  - URL: {base_url or 'None (Missing)'}\n")
    sys.stderr.write(f"  - Model: {model_name}\n\n")

    return {
        "API_BASE_URL":     base_url or "https://router.huggingface.co/v1", # Fallback ONLY for local, but URL check will warn
        "API_KEY":          api_key or "EMPTY",
        "MODEL_NAME":       model_name,
        "HF_TOKEN":         os.getenv("HF_TOKEN", ""),
        "LOCAL_IMAGE_NAME": os.getenv("LOCAL_IMAGE_NAME", "safe_station:latest"),
    }

def get_leaderboard_score(total_reward: float) -> float:
    """Official Meta leaderboard scoring logic (0.0 to 1.0)."""
    offset = 1000.0
    max_possible = 2000.0
    score = (total_reward + offset) / max_possible
    return max(0.0, min(0.99, score))

# =====================================================================
# SECTION 1: InferenceWrapper (Standard Gym-Style Interface)
# =====================================================================
class InferenceWrapper:
    def __init__(self, url: str = "http://localhost:8000"):
        self.client = SafeStationEnv(base_url=url)
        self.action_space = list(ACTIONS.keys())

    async def reset(self) -> Dict[str, float]:
        res = await self.client.reset()
        return _to_obs_dict(res.observation)

    async def step(self, action: int) -> Tuple[Dict[str, float], float, bool]:
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}.")
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
# =====================================================================

# =====================================================================
# SECTION 1.5: LLM Agent (Compliance Layer)
# =====================================================================
class LLMAgent:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        # Explicit warning if config is incomplete
        if not base_url or base_url == "https://router.huggingface.co/v1":
             sys.stderr.write("  [LLM AGENT] Warning: Using default HF Router. Ensure API_BASE_URL is set in environment.\n")
        
        self.client = OpenAI(base_url=base_url, api_key=api_key) if OpenAI else None
        self.model_name = model_name

    def get_action(self, obs: Dict[str, float]) -> int:
        """Asks the LLM for an action choice (0-3)."""
        if not self.client:
            sys.stderr.write("  [LLM AGENT] OpenAI client not initialized. Falling back to heuristics.\n")
            return heuristic_pick_action(obs["car_present"], obs["grid_price"], obs["station_battery_level"])

        prompt = (
            f"You are an AI managing an EV Charging Station. Decide the best action (0-3) based on state:\n"
            f"- Hour: {obs['hour']}\n"
            f"- Grid Price: {obs['grid_price']}\n"
            f"- Station Battery: {obs['station_battery_level']}%\n"
            f"- Car Present: {'YES' if obs['car_present'] else 'NO'}\n"
            f"- Car Need: {obs['car_battery_need']} units\n\n"
            f"Actions:\n"
            f"0: Wait / Grid Charge (Standard)\n"
            f"1: Battery Charge (Use BESS)\n"
            f"2: Top-Up BESS (Buy when cheap)\n"
            f"3: Hybrid Charge (Fast)\n\n"
            f"Respond ONLY with the action number (0, 1, 2, or 3)."
        )

        try:
            # Increased timeout to 15s to handle proxy latency
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
                timeout=15.0
            )
            content = response.choices[0].message.content.strip()
            
            # Simple digit extraction
            for char in content:
                if char in "0123":
                    return int(char)
            
            sys.stderr.write(f"  [LLM AGENT] Unexpected response format: '{content}'. Falling back to heuristics.\n")
            return heuristic_pick_action(obs["car_present"], obs["grid_price"], obs["station_battery_level"])

        except Exception as e:
            error_msg = str(e)
            sys.stderr.write(f"  [LLM ERROR] Step {obs.get('hour', '?')} call failed: {error_msg}\n")
            
            # Diagnostic hints for the user in the dashboard
            if "401" in error_msg:
                sys.stderr.write("  [HINT] 401 Unauthorized: Check if API_KEY is valid or correctly injected.\n")
            elif "404" in error_msg:
                sys.stderr.write("  [HINT] 404 Not Found: Check API_BASE_URL and MODEL_NAME.\n")
            elif "timeout" in error_msg.lower():
                sys.stderr.write("  [HINT] Request timed out. The proxy or model server is slow.\n")
                
            return heuristic_pick_action(obs["car_present"], obs["grid_price"], obs["station_battery_level"])

def heuristic_pick_action(car_present, grid_price, bess_level, car_need=0.0):
    if car_present:
        if bess_level >= 20.0:
            return 1 if grid_price > 10.0 else 3
        return 0
    else:
        if grid_price < 5.0 and bess_level < 100.0: return 2
        return 0

def compute_reward_math(action_id, grid_price, car_present, car_need_before, bess_before):
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

def run_full_episode_test(initial_hour, initial_bess, initial_car_present, initial_car_needs, m_vars):
    env = SafeStationEnvironment()
    env.reset()
    env.hour                  = initial_hour
    env.grid_price            = env._get_grid_price(initial_hour)
    env.station_battery_level = float(initial_bess)
    env.car_present           = int(initial_car_present)
    env.car_battery_need      = float(initial_car_needs)

    agent = LLMAgent(m_vars["API_BASE_URL"], m_vars["API_KEY"], m_vars["MODEL_NAME"])

    episode_reward = 0.0
    steps_taken = 0
    
    # Phase 2: Charging (Briefed for logs)
    for s in range(1, 11):
        if not env.car_present: break
        steps_taken = s
        
        obs = {
            "hour": env.hour,
            "grid_price": env.grid_price,
            "station_battery_level": env.station_battery_level,
            "car_present": env.car_present,
            "car_battery_need": env.car_battery_need
        }
        
        c_bef, n_bef, b_bef, gp = env.car_present, env.car_battery_need, env.station_battery_level, env.grid_price
        
        # ACTING via LLM Agent
        aid = agent.get_action(obs)
        
        env.step(SafeStationAction(action=aid))
        base, bonus, op, tp = compute_reward_math(aid, gp, c_bef, n_bef, b_bef)
        step_reward = (base + bonus + op + tp)
        episode_reward += step_reward
        
        # OpenEnv Structured Output
        print(f"[STEP] step={s} reward={step_reward:.4f}", flush=True)

    return episode_reward, steps_taken

# =====================================================================
# Main Validation Flow
# =====================================================================

async def main():
    # OpenEnv Structured Output Start
    print("[START] task=safe_station", flush=True)

    m_vars = get_mandatory_vars()
    # print(f"\n{SEP}\n  HACKATHON SYSTEM INITIALIZATION\n{SEP}")
    # print(f"  API_BASE_URL:     {m_vars['API_BASE_URL']}")
    # print(f"  MODEL_NAME:       {m_vars['MODEL_NAME']}")
    # print(f"  LOCAL_IMAGE_NAME: {m_vars['LOCAL_IMAGE_NAME']}")
    # print(f"{SEP}")

    # 1. API Compliance
    print("\n[Compliance] Connecting to environment server at http://localhost:8000...")
    try:
        env = InferenceWrapper()
        obs = await env.reset()
        print("  [PASS] reset() OK")
        await env.step(0)
        print("  [PASS] step()  OK")
        await env.close()
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 2. Physical Test
    rt = random.randint(0, 23)
    rb = float(random.randint(20, 100))
    rc = 1 # Force car present to ensure [STEP] blocks are printed
    rn = float(random.randint(20, 80)) if rc else 0.0
    
    total_reward, total_steps = run_full_episode_test(rt, rb, rc, rn, m_vars)
    
    # OpenEnv Structured Output End
    final_score = get_leaderboard_score(total_reward)
    print(f"[END] task=safe_station score={final_score:.4f} steps={total_steps}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
