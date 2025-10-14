import gymnasium
import numpy as np
import collections
import random

env = gymnasium.make("Blackjack-v1", render_mode="human")

#print(f"Observation Space: {env.observation_space}")
#print(f"Action Space: {env.action_space}")

state, info = env.reset()
done = False
total_reward = 0
print(f"State: {state}")

while not done:

    #action = env.action_space.sample
    action = int(input("Action (0: Stand, 1: Hit): "))
    #print(f"Action (0: Stand, 1: Hit): {action}")
    state, reward, terminated, truncated, info = env.step(action)
    print(f"State: {state}")
    done = terminated or truncated
    total_reward += reward
    env.render()

print(f"State: {state}")
print(f"Total reward: {total_reward}")

env.close()