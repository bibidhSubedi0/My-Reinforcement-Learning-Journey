import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random

# Defines the 2D world where the agents live
class TwoDWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 200}

    
    def __init__(self, render_mode=None, grid_size=5, n_agents=5, n_obstacles=3):
        self.grid_size = grid_size
        self.n_agents = n_agents  
        self.window_size = 800  
        self.render_mode = render_mode # Human nai hunxa
       
        # For now goal is to be in the region define by these coordinates
        self.goals = [[(0, 0), (grid_size-1, 10)]]

        # Observation space will be a box from 0 to gridsize -1
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(self.n_agents, 2),  # (agent, x/y)
            dtype=np.int32
        )

        # Action space will be [4,4,...,4], i.e. 4 possible actions for each agents
        self.action_space = spaces.MultiDiscrete([4] * self.n_agents)  

        # Window as in pygame wala window
        self.window = None

        # For time stuff, pygame wala nai
        self.clock = None

        # Position of all the agents
        self.agent_positions = None 


    # Reset all the agents to same position
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_positions = [(self.grid_size/2 -1,self.grid_size-1)]*self.n_agents
        self.agent_positions = np.array(self.agent_positions, dtype=np.int32)

        # [(),(),...,()] here top level [] is duplicated and () are copied by reference
        return self.agent_positions.copy(), {}

    # Helper to check if the agent is in the goal area
    def is_in_goal_area(self, pos):
        x, y = pos
        for (x1, y1), (x2, y2) in self.goals:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    # Take steps
    def step(self, actions):

        # Each of these things need to be list as they are different for each agent
        next_states = []
        rewards = []
        terminations = []
        truncations = []
        infos = [{} for _ in range(self.n_agents)]

        # Apply the action on agents position
        for idx, action in enumerate(actions):
            if action == 0 and self.agent_positions[idx][1] > 0:  # up
                self.agent_positions[idx][1] -= 1
            elif action == 1 and self.agent_positions[idx][1] < self.grid_size - 1:  # down
                self.agent_positions[idx][1] += 1
            elif action == 2 and self.agent_positions[idx][0] > 0:  # left
                self.agent_positions[idx][0] -= 1
            elif action == 3 and self.agent_positions[idx][0] < self.grid_size - 1:  # right
                self.agent_positions[idx][0] += 1

            obs = (self.agent_positions[idx][0], self.agent_positions[idx][1])

            terminated = any(
                goal[0][0] <= obs[0] <= goal[1][0] and goal[0][1] <= obs[1] <= goal[1][1]
                for goal in self.goals
            )
                
            reward = 10 if terminated else 0
            truncation = False

            next_states.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncation)
        return next_states, rewards, terminations, truncations, infos


    def render(self):
        self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self, time_passed=0):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        cell_size = self.window_size // self.grid_size
        
        # Draw the goal areas
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
                color = (200, 200, 200)
                for goal in self.goals:
                    (x_min, y_min), (x_max, y_max) = goal
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        color = (255, 0, 0)
                        break
                pygame.draw.rect(canvas, color, rect)

        # Draw each agent
        for pos in self.agent_positions:
            agent_rect = pygame.Rect(
                pos[1] * cell_size,
                pos[0] * cell_size,
                cell_size,
                cell_size
            )
            pygame.draw.rect(canvas, (0, 0, 255), agent_rect)  # Blue agent

        # Display the timer at the top-left corner of the screen
        font = pygame.font.SysFont("Arial", 24)
        time_text = font.render(f"Time: {time_passed:.2f} s", True, (0, 0, 0))
        canvas.blit(time_text, (10, 10)) 

        # Update the screen
        self.window.blit(canvas, (0, 0))
        pygame.display.update()

        # Cap the frame rate
        self.clock.tick(self.metadata["render_fps"])
