import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random

class TwoDWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=5): # 5x5
        self.grid_size = grid_size
        self.window_size = 500  # pixels
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(2,),
            dtype=np.int32
        )

        self.action_space = spaces.Discrete(4)  # left, right, up, down

        self.window = None
        self.clock = None
        self.agent_pos = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([random.randint(0,4), random.randint(0,4)], dtype=np.int32)

        if self.render_mode == "human":
            self._render_frame()

        return (self.agent_pos[0],self.agent_pos[1]), {}

    def step(self, action):
        if action == 0 and self.agent_pos[1] > 0:  # up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # right
            self.agent_pos[0] += 1

        obs = (self.agent_pos[0],self.agent_pos[1])
        terminated = obs == (2,2)
        reward = 1 if terminated else 0
        truncated = False
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info
    



    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        cell_size = self.window_size // self.grid_size
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
                color = (0, 0, 0) if np.array_equal([x, y], self.agent_pos) else (200, 200, 200)
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
