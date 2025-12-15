import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np

class OrientationWrapper(gym.RewardWrapper):

    def __init__(self, env, orientation_coefficient = 1.0):
        super().__init__(env)
        self.orientation_coefficient = orientation_coefficient

    def reward(self,reward):
        pitch = self.unwrapped.data.qpos[2]

        orientation_penalty = self.orientation_coefficient * (1-np.cos(pitch))

        modified_reward = reward - orientation_penalty

        return modified_reward
    
def custom_env(weight=1.0):
    env = gym.make("HalfCheetah-v5")

    env = OrientationWrapper(env,orientation_coefficient=weight)
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    #env = gym.wrappers.ClipAction(env)

    #env = gym.make_vec(env, num_envs=4)

    return env

LOG_DIR = "./ppo_halfcheetah_logs/"
MODEL_PATH = "halfcheetah_ppo_modified_reward"

customEnv = custom_env(weight=5.0)
observation, info = customEnv.reset()

#Baseline
model = PPO("MlpPolicy", customEnv, verbose=1, device='cuda',learning_rate = 3e-4, n_steps=2048,batch_size=64,gamma=0.99, tensorboard_log=LOG_DIR)

#Hyperparameters Edited
#model = PPO("MlpPolicy", env, verbose=1, device='cuda',learning_rate = 9e-4, n_steps=2048,batch_size=128,gamma=0.98, tensorboard_log=LOG_DIR)

#Hyperparameters 2
#model = PPO("MlpPolicy", env, verbose=1, device='cuda',learning_rate = 6e-4, n_steps=2048,batch_size=64,gamma=0.99, tensorboard_log=LOG_DIR)


model.learn(total_timesteps=1_000_000, tb_log_name = "PPO1_RUN_1_Modified_Reward")

model.save(MODEL_PATH)

customEnv.close()