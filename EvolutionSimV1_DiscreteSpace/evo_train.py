import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import pygame
import time
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Register the environment
register(
    id="TwoDWorldMulti-v0",
    entry_point="evo_env:TwoDWorldEnv",
)

agentsn = 1
grid_size = 10

# Create environment
env = gym.make("TwoDWorldMulti-v0", render_mode="human", grid_size=grid_size, n_agents=agentsn)


# For each indivisual agents
class QLearningAgent:
    
    def __init__(self, grid_size, alpha=0.5, gamma=0.95, epsilon=0.5):
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor 
        self.initial_epsilon = epsilon # Used for the decay equation
        self.epsilon = epsilon # Exploration rate
        self.min_epsilon = 0.1 # Minimum exploration rate
        self.decay_rate = 0.01 # Exploration decay
        
        # Initialize Q-table for all grid positions
        self.qtable = {}
        for i in range(grid_size):
            for j in range(grid_size):
                self.qtable[(i, j)] = [0, 0, 0, 0]  # 4 actions

    # Choose next action from current satate
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon: # Exploration
            return random.randint(0, 3)  
            
        else:                                   # Exploitation
            values = self.qtable[state]
            max_value = max(values)
            max_indices = [i for i, v in enumerate(values) if v == max_value]
            return random.choice(max_indices)

    # Update the q table based on the reward
    def update_q_table(self, state, action, reward, next_state, done):
        next_max = max(self.qtable[next_state])
        if not done:
            self.qtable[state][action] += self.alpha * (reward + self.gamma * next_max - self.qtable[state][action])
        else:
            self.qtable[state][action] += self.alpha * (reward - self.qtable[state][action])
        
    def get_parameters(self):
        return [self.alpha, self.gamma, self.epsilon, self.initial_epsilon]


# Train the agents independently for each epoch
def train_multi_agents(env,agents, n_agents, episodes):

    # An iteration for each agent
    count = 0
    for episode in range(episodes):
        # --------------------------------- Epesiode set up --------------------------------------
        start_time = time.time()
        states, _ = env.reset()
        states = [tuple(state) for state in states]
        done_flags = [False] * n_agents
        duration = time.time() - start_time


        # --------------------------------- Main Epesoide --------------------------------------
        # Main session of the epesoide 
        steps = 0
        while (duration<=0.01):  # Note at this time frame, the agents can make an avera of around 500 steps
            actions = []
            for i in range(n_agents):
                # if not done_flags[i]:
                action = agents[i].choose_action(states[i])
                steps +=1
                # else:
                #     action = None  # No action if already done
                actions.append(action)

            # Pass all actions to environment and get the rewards and other flag values
            next_states, rewards, terminations, truncations, infos = env.unwrapped.step(actions)
            
            # Update Q-tables
            for i in range(n_agents):
                if not done_flags[i]:
                    next_state = tuple(next_states[i])
                    agents[i].update_q_table(states[i], actions[i], rewards[i], next_state, terminations[i] or truncations[i])
                    states[i] = next_state
                    done_flags[i] = terminations[i] or truncations[i]
            duration = time.time() - start_time

        
        print("Steps taken : ",steps)


        # --------------------------------- After each epesiodes ----------------------------------

        # purge all the agents not in the zoneeee by deleting the instance of the agent and also removing from the environment
        # pos = []
        # pos_to_pop = []
        # for i in range(n_agents):
        #     if not env.unwrapped.is_in_goal_area(env.unwrapped.get_agent_position(i)):
        #         # actually delete the agents instance
        #         pos_to_pop.append(i)
        #         pos.append(env.unwrapped.get_agent_position(i))
        # pos_to_pop.sort(reverse=True)
        # for index in pos_to_pop:
        #   if 0 <= index < len(agents):  
        #         agents.pop(index)
        # n_agents = n_agents - len(pos)
        # env.unwrapped.remove_agents(n_agents,pos)

        # Decay epsilon
        for agent in agents:
            agent.epsilon = max(agent.min_epsilon, agent.initial_epsilon * pow(np.e, -1*0.01609*count))
            count +=1
            print(agent.get_parameters())

        print(f"Episode {episode + 1} finished.")
    return agents

def test_agent_from_random_starts(env, agent, n_tests=10):
    grid_size = env.unwrapped.grid_size
    visited_positions = set()

    for idx in range(n_tests):
        # Generate a unique random start position
        while True:
            start_pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            if start_pos not in visited_positions:
                visited_positions.add(start_pos)
                break

        # Reset and manually set agent position
        env.reset()
        env.unwrapped.agent_positions[0] = np.array(start_pos, dtype=np.int32)
        
        state = tuple(env.unwrapped.agent_positions[0])
        path = [state]
        done = False

        while not done:
            action = np.argmax(agent.qtable[state])  # greedy policy
            next_state, reward, terminated, truncated, _ = env.unwrapped.step([action])
            state = tuple(next_state[0])
            path.append(state)
            done = terminated[0] or truncated[0]
            env.unwrapped._render_frame()
        
        # Plot the path
        grid = np.zeros((grid_size, grid_size))
        for step_idx, (i, j) in enumerate(path):
            grid[i][j] = step_idx + 1

        plt.figure(figsize=(6, 6))
        sns.heatmap(grid, annot=True, fmt=".0f", cmap='coolwarm', cbar=False, linewidths=0.5, linecolor='gray')
        plt.title(f"Test {idx + 1}: Path from {start_pos}")
        plt.show()

def get_heat_map(agents):
    df = []
    actions = ['left', 'right', 'up', 'down']

    for agent in agents:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5)) 
        
        for action_idx in range(4):
            agent_values = []
            for key, val in agent.qtable.items():
                agent_values.append(val[action_idx])  

            agent_values = np.array(agent_values).reshape((grid_size, grid_size))

            sns.heatmap(agent_values, annot=True, cmap='YlGnBu', cbar=True, ax=axs[action_idx])
            axs[action_idx].set_title(f'Action: {actions[action_idx]}')

        plt.suptitle('Q-values Heatmap for All Actions', fontsize=16)
        plt.show()

def get_policy_arrows(agent, grid_size):
    # Define action → arrow direction mappings
    action_to_vector = {
        0: (-1, 0),  # Left
        1: (1, 0),   # Right
        2: (0, -1),  # Up
        3: (0, 1)    # Down
    }

    X, Y, U, V = [], [], [], []

    for i in range(grid_size):
        for j in range(grid_size):
            state = (i, j)
            if state in agent.qtable:
                best_action = np.argmax(agent.qtable[state])
                dx, dy = action_to_vector[best_action]
                X.append(j)
                Y.append(i)
                U.append(dx)
                V.append(dy)

    plt.figure(figsize=(grid_size, grid_size))
    plt.quiver(X, Y, U, V, scale=1, scale_units='xy', angles='xy', color='blue')
    plt.gca().invert_yaxis()
    plt.xticks(np.arange(0, grid_size))
    plt.yticks(np.arange(0, grid_size))
    plt.grid(True)
    plt.title("Policy Visualization: Arrows Indicate Best Actions")
    plt.show()


# Start training
n_agents = agentsn
episodes = 100
agents = [QLearningAgent(env.unwrapped.grid_size) for _ in range(n_agents)]
agents = train_multi_agents(env,agents, n_agents, episodes)

get_heat_map(agents)
get_policy_arrows(agents[0], grid_size)
test_agent_from_random_starts(env, agents[0], n_tests=3)



env.close()
