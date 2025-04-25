# q_learning_agent.py

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import pygame
import time
import random
import pandas as pd


# Register the environment defined previosuly
register(
    id="TwoDWorld-v0",
    entry_point="2D_world_env:TwoDWorldEnv",  # module:class
)
env = gym.make("TwoDWorld-v0", render_mode="human")



# Goal --- Make it to the bottom left square





class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.9, episodes=10):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.episodes = episodes

        x_max, y_max = 5, 5  # Assuming grid size of 5x5 for Q-table
        self.qtable = {}

        # Loop to initialize points from (0, 0) to (x_max, y_max)
        for i in range(x_max):  # Loop over x coordinates
            for j in range(y_max):  # Loop over y coordinates
                self.qtable[(i, j)] = [0, 0, 0, 0]  # Set values to [0, 0, 0, 0] (for 4 possible actions)

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return self.env.action_space.sample()
        
        else:
            # Exploitation: choose the action with the highest Q-value for the current state
            values = self.qtable[state]
            max_value = max(values)  # Find the maximum value
            max_index = values.index(max_value)  # Get the index of the maximum value
            return max_index
        
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

    def train(self):
        # Training loop for Q-learning
        for episode in range(self.episodes):
            print("Current : ",episode)

            state,_ = self.env.reset()  # Reset environment to initial state
            state = tuple(state)
            done = False

            while not done:
                # Choose action based on current state
                action = self.choose_action(state)  
                # Take action, observe next state and reward
                next_state, reward, done, trunc, info = self.env.step(action)  
                # Update Q-table
                self.update_q_table(state, action, reward, next_state, done)  
                # Move to next state
                state = next_state  


agent = QLearningAgent(env)
agent.train()

print(agent.qtable)
