"""
To play as human, run in terminal: flappy_bird_gymnasium
To see a random agent playing, run in terminal: flappy_bird_gymnasium --mode random
To see a Deep Q Network playing, run in terminal: flappy_bird_gymnasium --mode dqn
"""
# Import the flappy bird environment
import flappy_bird_gymnasium

# If env is compatible with gymnasium, copy this general pattern of code following...

import gymnasium

# When coding DQN, it is good to test on a simple environment.
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False) # not using LIDAR for this project

obs, _ = env.reset()

# Infinite loop, calling sample function on action space to get a random action and execute it, then get the results

while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Action space — 0 or 1. 0 is do nothing, 1 is flap.

    # Processing:
    # Step function executes the action.
    # # Gives observation of the next state,
    # the reward gained, termination status,
    # idk, and info for debugging.
    obs, reward, terminated, _, info = env.step(action)

    # Obs — there are 12 parameters. Each number is obs corresponded to one of them.
    # The numbers are normalized [-1, 1]
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()