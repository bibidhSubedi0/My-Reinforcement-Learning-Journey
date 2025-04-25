import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random

class TwoDWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=5, n_obstacles=3):
        self.isfirst = True
        self.grid_size = grid_size
        self.window_size = 500  # pixels
        self.render_mode = render_mode
        self.n_obstacles = n_obstacles

        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(2,),
            dtype=np.int32
        )

        self.action_space = spaces.Discrete(4)  # left, right, up, down

        self.goal = (grid_size // 2, grid_size // 2)
        self.avoid_cells = self._generate_avoid_cells()

        self.window = None
        self.clock = None
        self.agent_pos = None

    def _generate_avoid_cells(self):
        cells = set()
        while len(cells) < self.n_obstacles:
            cell = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if cell != self.goal:
                cells.add(cell)

        return list(cells)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            start_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if start_pos != self.goal and start_pos not in self.avoid_cells:
                break
        self.agent_pos = np.array(start_pos, dtype=np.int32)

        if self.render_mode == "human":
            self._render_frame()

        return (self.agent_pos[0], self.agent_pos[1]), {}

    def step(self, action):
        if action == 0 and self.agent_pos[1] > 0:  # up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # right
            self.agent_pos[0] += 1

        obs = (self.agent_pos[0], self.agent_pos[1])

        terminated = obs == self.goal
        reward = 10 if terminated else 0

        truncated = obs in self.avoid_cells
        if truncated:
            reward = -15  # Strong penalty

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
                color = (200, 200, 200)
                if (x, y) == tuple(self.agent_pos):
                    color = (100, 100, 0)
                elif (x, y) == self.goal:
                    color = (255, 0, 0)
                elif (x, y) in self.avoid_cells:
                    color = (0, 0, 0)
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
        
        if self.isfirst:
            pygame.display.update()
            
            # Capture the current screen and save it as an image
            pygame.image.save(pygame.display.get_surface(), "img.jpg")
            self.isfirst = False


