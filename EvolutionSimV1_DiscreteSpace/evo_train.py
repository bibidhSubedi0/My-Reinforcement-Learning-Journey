import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import pygame
import time
import random
import csv

# Register the environment
register(
    id="TwoDWorldMulti-v0",
    entry_point="evo_env:TwoDWorldEnv",  # <- assumes your modified env supports multiple agents
)

# Create environment
env = gym.make("TwoDWorldMulti-v0", render_mode="human", grid_size=40, n_agents=5)

class QLearningAgent:
    def __init__(self, grid_size, alpha=0.5, gamma=0.95, epsilon=1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.7
        self.decay_rate = 1
        
        # Initialize Q-table for all grid positions
        self.qtable = {}
        for i in range(grid_size):
            for j in range(grid_size):
                self.qtable[(i, j)] = [0, 0, 0, 0]  # 4 actions

    def choose_action(self, state):
        # if random.uniform(0, 1) < self.epsilon:
        return random.randint(0, 3)  # Random action
        # else:
        #     values = self.qtable[state]
        #     max_value = max(values)
        #     max_indices = [i for i, v in enumerate(values) if v == max_value]
        #     return random.choice(max_indices)

    def update_q_table(self, state, action, reward, next_state, done):
        next_max = max(self.qtable[next_state])
        if not done:
            self.qtable[state][action] += self.alpha * (reward + self.gamma * next_max - self.qtable[state][action])
        else:
            self.qtable[state][action] += self.alpha * (reward - self.qtable[state][action])

def train_multi_agents(env, n_agents, episodes):
    agents = [QLearningAgent(env.unwrapped.grid_size) for _ in range(n_agents)]
    episode_durations = []

    for episode in range(episodes):
        start_time = time.time()

        # Reset environment and get initial states for all agents
        states, _ = env.reset()
        states = [tuple(state) for state in states]
        done_flags = [False] * n_agents

        while not all(done_flags):
            actions = []
            for i in range(n_agents):
                # if not done_flags[i]:
                action = agents[i].choose_action(states[i])
                # else:
                    # action = None  # No action if already done
                actions.append(action)

            # Pass all actions to environment
            next_states, rewards, terminations, truncations, infos = env.unwrapped.step(actions)

            # Update Q-tables
            for i in range(n_agents):
                if not done_flags[i]:
                    next_state = tuple(next_states[i])
                    agents[i].update_q_table(states[i], actions[i], rewards[i], next_state, terminations[i] or truncations[i])
                    states[i] = next_state
                    done_flags[i] = terminations[i] or truncations[i]

        duration = time.time() - start_time
        episode_durations.append((episode + 1, duration))

        # Decay epsilon
        for agent in agents:
            agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.decay_rate)

        print(f"Episode {episode + 1} finished.")

    # Save training times
    with open("episode_times.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "duration_seconds"])
        writer.writerows(episode_durations)

    return agents

# Start training
n_agents = 5
episodes = 1
agents = train_multi_agents(env, n_agents, episodes)

env.close()
