import random
import gymnasium as gym # type: ignore
import numpy as np # type: ignore

env = gym.make('Taxi-v3', render_mode=None)

alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100

q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        state = next_state
        if done or truncated:
            break
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Enable rendering to visualize the results
env = gym.make('Taxi-v3', render_mode='human')

for episode in range(5):
    state, _ = env.reset()
    done = False
    while not done:
        action = choose_action(state)
        state, reward, done, truncated, _ = env.step(action)
        try:
            env.render()
        except pygame.error:
            print("Pygame window closed unexpectedly.")
            break