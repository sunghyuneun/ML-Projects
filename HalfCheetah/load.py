import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("HalfCheetah-v5", render_mode = "human")

try:
    model = PPO.load("halfcheetah_ppo_modified_reward", env=env)
except FileNotFoundError:
    print("lol file is not here bro")
    exit()

print("model found.")

observation, info = env.reset()
episode_reward = 0

for step in range(5000):
    action, _ = model.predict(observation, deterministic = True)

    observation, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    if terminated or truncated:
        print(f"Episode finished after {step+1} steps. Reward: {episode_reward}")
        observation, info = env.reset()
        episode_reward = 0

env.close()