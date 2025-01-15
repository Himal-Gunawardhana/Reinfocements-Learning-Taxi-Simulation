import random
import gym
import numpy as np

env = gym.make('Taxi-v3')

alpha = 0.9     # how percentage is important new rewards(learning rate)
gamma = 0.95    # how important future rewards are (discount factor)
epsilon = 1.0   # all the actions in the beginning will be random and 
# if epsilon is zero that means always depending on q-values intead of random(exploration rate)
epsilon_decay = 0.9995  # how fast epsilon will decay
min_epsilon = 0.01  # minimum value of epsilon (at least 1% of the randomness)
num_episodes = 10000
max_steps = 100

q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()    # take random action
    else:
        return np.argmax(q_table[state, :])     # take action with highest q-value

for episode in range(num_episodes):     #training loop
    state, _ = env.reset()      # reset the environment

    done = False

    for step in range(max_steps):    # maximum 100 steps taken to reach the goal randomly while training
        action = choose_action(state)       # choose an action

        next_state, reward, done, truncated, info = env.step(action)    # take the action

        old_value = q_table[state, action]   # old q-value
        next_max = np.max(q_table[next_state, :])   # next state max q-value

        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)      # update q-value

        state = next_state   # update state

        if done or truncated:   # if done or truncated break the loop
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env = gym.make('Taxi-v3', render_mode='human')

for episode in range(5):
    state, _ = env.reset()
    done = False

    print('Episode', episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        if done or truncated:
            env.render()
            print('Finished episode', episode, 'with reward', reward)
            break

env.close()