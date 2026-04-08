#!/usr/bin/env python3
import asyncio
import os
from typing import List

# Mock OpenAI client since we don't have real keys here
class MockOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def get_model_message(self, step, obs, history):
        # A simple hardcoded agent that follows heuristic constraints
        # 0: Grid, 1: Battery, 2: Top-up, 3: Hybrid
        
        car_present = obs.car_present
        grid_price = obs.grid_price
        bess = obs.station_battery_level
        
        if car_present:
            # Need to charge the car
            if grid_price > 10.0 and bess >= 20.0:
                return 1 # Battery charge (Peak shave)
            if bess >= 20.0:
                return 3 # Hybrid (fastest)
            return 0 # Grid fallback
        else:
            # Car not present
            if grid_price < 5.0 and bess < 80.0:
                return 2 # Top-up (Strategic)
            return 0 # Wait/Do nothing mostly if we do action 0 when car not present it just costs nothing and progresses time

class MockOpenAIClient:
    def __init__(self, *args, **kwargs):
        pass

async def main() -> None:
    from client import SafeStationEnv
    from models import SafeStationAction
    
    # Normally read from env vars
    API_BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
    API_KEY = os.getenv("OPENAI_API_KEY", "test-key")
    IMAGE_NAME = "safe_station:latest"
    
    # We will connect locally to the server or start container.
    # For baseline, we connect to local server if running, or use docker.
    env = SafeStationEnv(base_url="http://localhost:8000")
    
    agent = MockOpenAI()

    for task_name in ["Task 1: Easy Start", "Task 2: Medium Operations", "Task 3: Hard Constraints"]:
        print(f"\n--- Starting {task_name} ---")
        history: List[str] = []
        rewards: List[float] = []

        try:
            result = await env.reset()
            obs = result.observation
            last_reward = 0.0
            
            for step in range(1, 48): # Max 48 steps to keep baseline fast
                if result.done:
                    break

                action_choice = agent.get_model_message(step, obs, history)
                action = SafeStationAction(action=action_choice)
                
                result = await env.step(action)
                obs = result.observation
                
                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                last_reward = reward

                history.append(f"Step {step}: Action {action_choice} -> reward {reward:+.2f} | BESS: {obs.station_battery_level}")

                if done:
                    break

            total_reward = sum(rewards)
            # Theoretical max is ~+300 per successful car service.
            max_possible = 1000.0 if "Easy" in task_name else 2000.0
            score = max(0.0, min(1.0, (total_reward + 1000) / max_possible))
            print(f"{task_name} completed in {step} steps. Total Reward: {total_reward:+.2f}. Score: {score:.2f}/1.0")

        except Exception as e:
            print(f"Episode failed: {e}")

    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
