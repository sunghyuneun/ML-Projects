import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("HalfCheetah-v5")
observation, info = env.reset()

model = PPO("MlpPolicy", env, verbose=1, device='cuda',n_steps=2048,batch_size=64,gamma=0.99)

print(f"Observation Space: {env.observation_space}]\n Action Space: {env.action_space}")

'''
for _ in range(5000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
'''
model.learn(total_timesteps=1_000_000)

model.save("halfcheetah_ppo_v5")

env.close()