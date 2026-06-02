"""
To play as human, run in terminal: flappy_bird_gymnasium
To see a random agent playing, run in terminal: flappy_bird_gymnasium --mode random
To see a Deep Q Network playing, run in terminal: flappy_bird_gymnasium --mode dqn
"""

import torch
from torch import nn
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)
        
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None # NN optimizer, initialize later

    def run(self, is_training=True, render=False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        # Test code on an easier environment first
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        rewards_per_episode = []
        epsilon_history = [self.epsilon_init]
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size) # we'll make it dynamic later

            epsilon = self.epsilon_init

            target_dqn = DQN(num_staes, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Track num steps taken for syncing target and policy networks
            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)


        for episode in itertools.count(): # train indefinitely, stop when we want to
            state, _ = env.reset()
            # convert state to a tensor
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() # explore
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.inference_mode(): # turn off gradient calculation to save on processing power
                        # tensor ([1, 2, 3, ..]) -> tensor([[1, 2, 3, ...]]) because dim=0 is batch dim
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze(dim=0).argmax() # get the action with largest Q value
                        # doing this bc the indices line up with the action space (0 is 0, 1 is 1)

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                # Accumulate reward for this episode
                episode_reward += reward

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1
                
                # move to new state
                state = new_state
            
            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            # if enough experiences have been collected
            if len(memory) > self.mini_batch_size:
                # Sample from memory
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Copy policy network to target network after a certain num of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
                    
def optimize(self, mini_batch, policy_dqn, target_dqn):
    # Tranpose the list of experiences and separate each element
    states, actions, new_states, rewards, terminations = zip(*mini_batch)

    # Stack tensors to create batch tensors
    # tensor ([[1, 2, 3]]) -> tensor([[1, 2, 3, ...], [4, 5, 6, ...], ...])
    states = torch.stack(states)
    actions = torch.stack(actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)
    terminations = torch.tensor(terminations).float().to(device)

    with torch.inference_mode():
        # Calculate target q values (expected return)
        target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

    # Calculate q values from current policy network
    current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)

    # Compute loss for the whole minibatch
    loss = self.loss_fn(current_q, target_q)

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


if __name__ == '__main__':
    agent = Agent('cartpole1')
    agent.run(is_training=True, render=True)

    