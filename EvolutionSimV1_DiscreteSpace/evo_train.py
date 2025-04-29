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
    entry_point="evo_env:TwoDWorldEnv",
)

agentsn = 100

# Create environment
env = gym.make("TwoDWorldMulti-v0", render_mode="human", grid_size=75, n_agents=agentsn)

# For each indivisual agents
class QLearningAgent:
    
    def __init__(self, grid_size, alpha=0.5, gamma=0.95, epsilon=0.5):
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor 
        self.epsilon = epsilon # Exploration rate
        self.min_epsilon = 0.7 # Minimum exploration rate
        self.decay_rate = 1 # Exploration decay
        
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
        
        


# Train the agents independently for each epoch
def train_multi_agents(env,agents, n_agents, episodes):
    # Initlize n_agents -> Will need to modify it later for to 
    # include agents from previous generations and reprodudce new agents
    # by replicating the q tables upto some factor

    
    # An iteration for each agent
    for episode in range(episodes):
        start_time = time.time()

        # Reset environment and get initial states for all agents
        states, _ = env.reset()
        states = [tuple(state) for state in states]
        done_flags = [False] * n_agents


        duration = time.time() - start_time

        while (duration<=25):
            actions = []
            for i in range(n_agents):
                # if not done_flags[i]:
                action = agents[i].choose_action(states[i])
                # else:
                #     action = None  # No action if already done
                actions.append(action)

            # Pass all actions to environment
            next_states, rewards, terminations, truncations, infos = env.unwrapped.step(actions)
            env.unwrapped._render_frame(time.time() - start_time)


            # Update Q-tables
            for i in range(n_agents):
                if not done_flags[i]:
                    next_state = tuple(next_states[i])
                    agents[i].update_q_table(states[i], actions[i], rewards[i], next_state, terminations[i] or truncations[i])
                    states[i] = next_state
                    done_flags[i] = terminations[i] or truncations[i]
                
            
            duration = time.time() - start_time

        # Now after requred duration purge all the agents not in the zoneeee
        pos = []
        for i in range(n_agents):
            if not env.unwrapped.is_in_goal_area(env.unwrapped.get_agent_position(i)):
                # How do i purge it hummm
                print(env.unwrapped.get_agent_position(i))
                pos.append(env.unwrapped.get_agent_position(i))
        
        
            
        

        n_agents = n_agents - len(pos)
        env.unwrapped.remove_agents(n_agents,pos)


        # Lemme see the q table of rest of the agents
        for i in range(n_agents):
            print(agents[i].qtable)



        # Decay epsilon
        for agent in agents:
            agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.decay_rate)

        print(f"Episode {episode + 1} finished.")



    return agents



# Start training
n_agents = agentsn
episodes = 50
agents = [QLearningAgent(env.unwrapped.grid_size) for _ in range(n_agents)]
agents = train_multi_agents(env,agents, n_agents, episodes)

env.close()
