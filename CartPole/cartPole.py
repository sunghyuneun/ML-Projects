import gymnasium
import numpy as np
from collections import defaultdict
import random

env = gymnasium.make('CartPole-v1')

#? Maybe Delete
observation, info = env.reset(seed=42)
terminated = False
truncated = False

#Position, Velocity, Angle, Angular Velocity Bins
POS_BINS = np.linspace(-4.8,4.8,9)
VEL_BINS = np.linspace(-3.5,3.5,9)
ANG_BINS = np.linspace(-.418,.418,9)
ANG_VEL_BINS = np.linspace(-5.0,5.0,9)

#Initialize Q Table, Q_table[discrete_state,action] is Q(s,a)
Q_table = defaultdict(lambda: np.zeros(env.action_space.n))

#Epsilon and Decay rates, Epsilon is exploration rate
epsilon = 1.0
EPSILON_DECAY_RATE = 0.99995
MIN_EPSILON = 0.01

#Q-Learning Update, Alpha is Learning Rate, Gamma is Discount rate
ALPHA = 0.1
DISCOUNT_RATE = .99


#print(f"Observation Space: {env.observation_space}")
#print(f"Action Space (0=Left, 1=Right): {env.action_space}")

MAX_EPISODES = 50000

#Discretizes the state based on the (global) bins
def discretize_state(state):
    pos, vel, ang, ang_vel = state
    
    discrete_pos = np.digitize(pos,POS_BINS)
    discrete_vel = np.digitize(vel, VEL_BINS)
    discrete_ang = np.digitize(ang, ANG_BINS)
    discrete_ang_vel = np.digitize(ang_vel, ANG_VEL_BINS)

    return (discrete_pos, discrete_vel, discrete_ang, discrete_ang_vel)


#Selects action based on epsilon
def select_action(discrete_state, Q_table, epsilon):

    #If below epsilon, do random
    if random.random() < epsilon:
        return env.action_space.sample()
    #Otherwise, return the action that provided maximum value at a state. 
    else:
        return np.argmax(Q_table[discrete_state])
    
def update_q_table(s, a, r, s_prime, done, Q_table):


    #Finding the TD target
    if done:
        #If terminal state, no future state. Target is just r
        max_future_q = 0
    else:
        # Q-Learning Off-Policy: Use max Q-Value of next state. 
        max_future_q = np.max(Q_table[s_prime])

    td_target = r + DISCOUNT_RATE * max_future_q

    #TD Error:

    td_error = td_target - Q_table[s][a]
    
    Q_table[s][a] += ALPHA * td_error

reached_truncate = False

#Training Episodes
for episode in range(1, MAX_EPISODES+1):
    first_state, _ = env.reset()
    current_discrete_state = discretize_state(first_state)
    done = False
    score = 0

    while not done:
        action = select_action(current_discrete_state, Q_table, epsilon)

        state, reward, terminated, truncated, info = env.step(action)
        if truncated:
            if not reached_truncate:
                print("First Truncation")
                reached_truncate = True

        done = terminated or truncated

        new_discrete_state = discretize_state(state)

        update_q_table(current_discrete_state, action, reward, new_discrete_state, done, Q_table)

        current_discrete_state = new_discrete_state
        score += reward
        
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY_RATE)
    if episode % 5000 == 0:
        print(f"Episode: {episode}, Score: {score}")

env.close()


last_env = gymnasium.make('CartPole-v1',render_mode = 'human')

total_reward = 0
while total_reward < 150:
    first_state, _ = last_env.reset()
    current_discrete_state = discretize_state(first_state)
    done = False
    total_reward = 0

    while not done:
        action = select_action(current_discrete_state, Q_table, 0)

        state, reward, terminated, truncated, info = last_env.step(action)
        done = terminated or truncated

        new_discrete_state = discretize_state(state)

        current_discrete_state = new_discrete_state
        total_reward += reward
        last_env.render()

print(f"\nFinished training. Total Reward: {total_reward}")

last_env.close()
