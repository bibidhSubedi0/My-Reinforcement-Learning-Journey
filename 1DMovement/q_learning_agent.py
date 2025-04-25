import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import pygame
import time


# Register the environment defined previosuly
register(
    id="LineWorld-v0",
    entry_point="line_world_env:LineWorldEnv",  # module:class
)
env = gym.make("LineWorld-v0", render_mode="human")


q_table = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.5    # Exploration rate

episodes = 100   # Total no. of times the agent goes through a full cycle

for episode in range(episodes):
    # First ma always reset the environment
    state, _ = env.reset()
    done = False

    print("Episdoe : ",episode)

    
    while not done:
        # Exploration v/s exploitation
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit


        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        best_next = np.max(q_table[next_state])
        q_table[state, action] += alpha * (reward + gamma * best_next - q_table[state, action])
        state = next_state



print("✅ Q-Table learned:")
print(q_table)






# # --- Visualization after training ---
# print("\n🎬 Running one episode with learned Q-table...\n")
# time.sleep(1)
# state, _ = env.reset()
# done = False

# while not done:
#     action = np.argmax(q_table[state])  # Exploitation only
#     next_state, reward, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated
#     state = next_state
#     time.sleep(0.5)
#     env.render()

# print("✅ Agent reached goal using learned Q-table.")
# pygame.time.wait(1000)
# pygame.quit()