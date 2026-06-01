"""
To play as human, run in terminal: flappy_bird_gymnasium
To see a random agent playing, run in terminal: flappy_bird_gymnasium --mode random
To see a Deep Q Network playing, run in terminal: flappy_bird_gymnasium --mode dqn
"""

import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def run(self, is_training=True, render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions).to_device(device)

        obs, _ = env.reset()

        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Action space — 0 or 1. 0 is do nothing, 1 is flap.

            # Processing:
            obs, reward, terminated, _, info = env.step(action)

            # Checking if the player is still alive
            if terminated:
                break

        env.close()

if __name__ == '__main__':
    agent = Agent()
    agent.run(
        is_training=True,
        render=True)
    