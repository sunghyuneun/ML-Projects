import gymnasium

env = gymnasium.make('CartPole-v1', render_mode="human")

observation, info = env.reset(seed=42)
terminated = False
truncated = False
total_reward = 0

print(f"Observation Space: {env.observation_space}")
print(f"Action Space (0=Left, 1=Right): {env.action_space}")

while not terminated and not truncated:
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

print(f"\nEpisode finished. Total Reward: {total_reward}")

env.close()