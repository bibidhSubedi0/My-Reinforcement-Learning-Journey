import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class LineWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.window = None
        self.window_size = 300
        self.steps_taken = 0

        self.num_cells = 6
        self.agent_pos = 0

        # Two actions: 0 = Left, 1 = Right
        self.action_space = spaces.Discrete(2)

        # Observation = agent position (0 to 4)
        self.observation_space = spaces.Discrete(self.num_cells)

    def reset(self, seed=None, options=None):
        self.steps_taken = 0
        self.agent_pos = 0
        if self.render_mode == "human":
            self._render_frame()
        return self.agent_pos, {}

    def step(self, action):
        
        self.steps_taken+=1

        if action == 0 and self.agent_pos > 0:
            self.agent_pos -= 1
        elif action == 1 and self.agent_pos < self.num_cells - 1:
            self.agent_pos += 1

        reward = 1.0 if self.agent_pos == self.num_cells - 1 else 0.0
        terminated = self.agent_pos == self.num_cells - 1

        # truncated = self.steps_taken == 20
        truncated = False

        # if self.render_mode == "human":
        #     self._render_frame()

        return self.agent_pos, reward, terminated, truncated, {}
    


    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, 100))

        self.window.fill((255, 255, 255))  # white background

        # Draw 5 cells
        cell_width = self.window_size // self.num_cells
        for i in range(self.num_cells):
            color = (0, 0, 0) if i == self.agent_pos else (200, 200, 200)
            pygame.draw.rect(self.window, color, pygame.Rect(i * cell_width, 20, cell_width - 2, 60))

        pygame.display.flip()

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
