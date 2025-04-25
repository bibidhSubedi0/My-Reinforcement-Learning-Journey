# q_learning_agent.py

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import pygame
import time
import random
import pandas as pd
import csv
import time


# Register the environment defined previosuly
register(
    id="TwoDWorld-v0",
    entry_point="2D_world_env:TwoDWorldEnv",  # module:class
)
env = gym.make("TwoDWorld-v0", render_mode="human", grid_size=8)


class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=1, episodes=1000):
        self.min_epsilon = 0.1
        self.decay_rate = 0.995
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.episodes = episodes

        x_max, y_max = env.unwrapped.grid_size, env.unwrapped.grid_size
        self.qtable = {}

        # Loop to initialize points from (0, 0) to (x_max, y_max)
        for i in range(x_max):  # Loop over x coordinates
            for j in range(y_max):  # Loop over y coordinates
                self.qtable[(i, j)] = [0, 0, 0, 0]  # Set values to [0, 0, 0, 0] (for 4 possible actions)
        
        print(self.qtable)

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return self.env.action_space.sample()
        
        else:
            # Exploitation: choose the action with the highest Q-value for the current state
            values = self.qtable[state]
            max_value = max(values)
            max_indices = [index for index, value in enumerate(values) if value == max_value]
            return random.choice(max_indices)  # Randomly pick one of the max-value indices
        
    def update_q_table(self, state, action, reward, next_state, done):
        next_state = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state

        # Bellman equation for Q-learning
        if not done:
            # Update Q-value using the formula:
            # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
            next_max = max(self.qtable[next_state])  # Get max Q-value for next state
            self.qtable[state][action] += self.alpha * (reward + self.gamma * next_max - self.qtable[state][action])
        else:
            # If done, just update Q-value with reward
            self.qtable[state][action] += self.alpha * (reward - self.qtable[state][action])
    
    
    def save_episode_times(self, durations, filename="episode_times.csv"):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "duration_seconds"])
            writer.writerows(durations)


    def train(self):
        episode_durations = []
        # Training loop for Q-learning
        for episode in range(self.episodes):
            start_time = time.time()

            state,_ = self.env.reset()  # Reset environment to initial state
            state = tuple(state)
            done = False
            trunc = False

            r = False
            while not (done or trunc):
                # Choose action based on current state
                action = self.choose_action(state)  
                # Take action, observe next state and reward
                next_state, reward, done, trunc, info = self.env.step(action)  

                r= reward
                # Update Q-table
                self.update_q_table(state, action, reward, next_state, done)  
                # Move to next state
                state = next_state  
            
            
            print("Current : ",episode)
            duration = time.time() - start_time
            episode_durations.append((episode + 1, duration))
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

        
        self.save_episode_times(episode_durations)


agent = QLearningAgent(env)
agent.train()



def save_q_table_dict_to_csv(q_table, filename="q_table.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "p1", "p2", "p3", "p4"])  # header
        
        for (x, y), values in q_table.items():
            row = [x, y] + values
            writer.writerow(row)

save_q_table_dict_to_csv(agent.qtable)

# Do a best possible run 
# Do a best possible run 
print("\nRunning Best Policy...")
time.sleep(1)

total_reward = 0
steps = 0


for i in range(10):

    state, _ = env.reset()
    state = tuple(state)
    done = False
    trunc = False
    while not (done or trunc):
        env.render()

        # Pick the best action (greedy)
        q_values = agent.qtable[state]
        best_action = int(np.argmax(q_values))

        next_state, reward, done, trunc, info = env.step(best_action)
        state = tuple(next_state)
        total_reward += reward
        steps += 1

print(f"\nfinished best run in {steps} steps with total reward: {total_reward}")
env.close()

print(agent.qtable)
