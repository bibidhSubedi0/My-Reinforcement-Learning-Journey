import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class Continuous2DEnv(gym.Env):
    def __init__(self):
        super(Continuous2DEnv, self).__init__()

        # Define the observation space: 2D position (x, y), continuous values between -10 and 10
        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0]), high=np.array([10.0, 10.0]), dtype=np.float32)

        # Define the action space: 2D velocity (vx, vy), continuous values between -1 and 1
        self.action_space = spaces.Box(low=np.array([-0.1, -0.1]), high=np.array([0.1, 0.1]), dtype=np.float32)

        pygame.init()

        # Screen setup
        self.screen_width = 500
        self.screen_height = 500
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.state = np.random.uniform(low=-10.0, high=10.0, size=(2,))

        self.scale = 25  # scale the world coordinates to fit in the screen size
        
        # Define a color for the agent
        self.agent_color = (0, 0, 255)  # Blue agent

    def reset(self):
        # Reset the agent to a random position
        self.state = np.random.uniform(low=-10.0, high=10.0, size=(2,))
        self._render()  # render the initial state
        return self.state, {}

    def step(self, action):
        # Extract action (velocity in x and y)
        vx, vy = action
        
        # Update position based on velocity (simple integration step)
        self.state += np.array([vx, vy])
        
        # Clip the position to the environment's limits
        self.state = np.clip(self.state, -10.0, 10.0)
        
        # For simplicity, we assume the agent always gets a small negative reward (could be modified for goals)
        reward = -0.1
        
        # Check if the episode is done (for now, we don't define any terminal condition)
        done = False
        info = {}

        # Render the environment for visualization
        self._render()
        
        return self.state, reward, done, False, info

    def _render(self):
        # Convert the agent's position to screen coordinates
        x_pos, y_pos = self.state
        screen_x = int(self.screen_width / 2 + x_pos * self.scale)
        screen_y = int(self.screen_height / 2 - y_pos * self.scale)

        # Fill the screen with a white background
        self.screen.fill((255, 255, 255))

        # Draw the agent (as a blue dot)
        pygame.draw.circle(self.screen, self.agent_color, (screen_x, screen_y), 10)

        # Update the display
        pygame.display.flip()

        # Delay to make the update visible
        self.clock.tick(60)

    def close(self):
        pygame.quit()

# Usage example
if __name__ == "__main__":
    env = Continuous2DEnv()

    # Initialize the environment
    obs, _ = env.reset()

    # Define a random action sequence to test
    while True:
        # Random action: (vx, vy)
        action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, done, truncated, info = env.step(action)

        # Check for any user inputs to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

    env.close()
