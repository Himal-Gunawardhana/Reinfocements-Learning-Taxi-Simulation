import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

class CustomTaxiEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.initial_taxi_row = None
        self.initial_taxi_col = None
        self.initial_passenger_loc = None
        self.initial_destination = None

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed)
        
        # Force set the initial state
        if all(x is not None for x in [self.initial_taxi_row, self.initial_taxi_col, 
                                     self.initial_passenger_loc, self.initial_destination]):
            
            # Directly modify the underlying environment state
            self.unwrapped.s = self.unwrapped.encode(
                self.initial_taxi_row,
                self.initial_taxi_col,
                self.initial_passenger_loc,
                self.initial_destination
            )
            
            # Update the state to reflect our changes
            state = self.unwrapped.s
            
        return state, info

def run(episodes, is_training=True, render=False, initial_config=None):
    env = CustomTaxiEnv(gym.make('Taxi-v3', render_mode='human' if render else None))
    
    # Set initial configuration if provided
    if initial_config:
        env.initial_taxi_row = initial_config['taxi_row']
        env.initial_taxi_col = initial_config['taxi_col']
        env.initial_passenger_loc = initial_config['passenger_loc']
        env.initial_destination = initial_config['destination']

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = rewards

    env.close()

    # Plot rewards
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')

    if is_training:
        with open("taxi.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    # Define initial configuration
    # Locations: 0=Red, 1=Green, 2=Yellow, 3=Blue
    initial_config = {
        'taxi_row': 0,        # 0-4 (top to bottom)
        'taxi_col': 0,        # 0-4 (left to right)
        'passenger_loc': 3,   # 0=Red location
        'destination': 2      # 2=Yellow location
    }

    # Training phase
    print("Starting training...")
    run(15000, is_training=True, initial_config=initial_config)

    # Evaluation phase with rendering
    print("Starting evaluation...")
    run(1, is_training=False, render=True, initial_config=initial_config)