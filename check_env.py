from cityflow_env import CityFlowEnv
import numpy as np

print("Creating the environment...")
env = CityFlowEnv("examples/config.json")

# 2. Reset environment
obs, info = env.reset()
print(f"Start state (num of cars): {obs}")
print(f"State dimension: {obs.shape}")

# 3. Random 5 steps for test
print("\n Test begin...")
for i in range(5):
    # Choose an action randomly (0-7)
    action = env.action_space.sample()
    
    # Execute movement
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {i+1}: Action={action}, Reward={reward}, New state={next_obs[:3]}...")

print("\n Gym test pass!")