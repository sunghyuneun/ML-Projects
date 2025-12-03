import gymnasium as gym

env = gym.make("HalfCheetah-v5", render_mode="human")
observation, info = env.reset()

print(f"Observation Space: {env.observation_space}]\n Action Space: {env.action_space}")


for _ in range(5000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()


env.close()