"""
To play as human, run in terminal: flappy_bird_gymnasium
To see a random agent playing, run in terminal: flappy_bird_gymnasium --mode random
To see a Deep Q Network playing, run in terminal: flappy_bird_gymnasium --mode dqn
"""

import torch
from torch import nn
import flappy_bird_gymnasium
import gymnasium
from gymnasium.spaces import Discrete
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import itertools
import numpy as np
from torch.optim import Optimizer

# for printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Dir for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # for now, model is small and env is simple

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)
        self.hyperparameter_set = hyperparameter_set
        
        self.env_id = hyperparameters['env_id']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']

        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {}) # get optional ev specific params
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        
        self.loss_fn = nn.MSELoss()
        self.optimizer: Optimizer | None = None  # init later at line 58

        # Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        # Timer variable stuff
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: {'Training' if is_training else 'Testing'} starting"
        print(log_message, flush=True)
        if is_training:
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
        else:
            print(log_message, flush=True)  # already printed above; could skip the file write entirely

        # Create instance of the environment
        # pass in **self.env_make_params to get env-specific parameters from hyperparameters.yml
        # env = gymnasium.make(self.env_id, render_mode="human" if render else None, use_lidar=False, **self.env_make_params)
        # Test code on an easier environment first
        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
    
        # num of states is the dim of the input layer
        assert env.observation_space.shape is not None
        num_states = env.observation_space.shape[0]

        # num possible actions
        assert isinstance(env.action_space, Discrete)
        num_actions = int(env.action_space.n)

        # list to keep track of rewards per episode
        rewards_per_episode = []

        # Create policy network
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            # init epsilon
            epsilon = self.epsilon_init

            memory = ReplayMemory(self.replay_memory_size) # we'll make it dynamic later

            # Create target network and make it identical to policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Create policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # Create epsilon history to keep track of decay
            epsilon_history = [self.epsilon_init]

            # Track num steps taken for syncing target and policy networks
            step_count = 0

            # Track best reward
            best_reward = float('-inf')
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        for episode in itertools.count(): # train indefinitely, stop when we want to
            state, _ = env.reset()
            # convert state to a tensor
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            truncated = False
            episode_reward = 0.0

            while (not terminated and not truncated and episode_reward < self.stop_on_reward):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() # explore
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.inference_mode(): # turn off gradient calculation to save on processing power
                        # tensor ([1, 2, 3, ..]) -> tensor([[1, 2, 3, ...]]) because dim=0 is batch dim
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze(dim=0).argmax() # get the action with largest Q value
                        # doing this bc the indices line up with the action space (0 is 0, 1 is 1)

                # Processing:
                new_state, reward, terminated, truncated, _ = env.step(action.item()) # _ is info

                # Accumulate reward for this episode
                episode_reward += float(reward)

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience to memory
                    memory.append((state, action, new_state, reward, terminated))
                    
                    # Increment step counter
                    step_count += 1
                
                # move to new state
                state = new_state
            
            # Track rewards collected per episode
            rewards_per_episode.append(episode_reward)

            # Save model when best reward is achieved
            if is_training:
                if episode_reward > best_reward:
                    if best_reward == float('-inf') or best_reward == 0:
                        pct_change = float('inf')
                    else:
                        pct_change = (episode_reward - best_reward) / abs(best_reward) * 100
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Episode {episode}: New best reward {episode_reward:0.1f} ({pct_change:+.1f}%)"
                    print(log_message, flush=True)

                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if (current_time - last_graph_update_time) > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # if enough experiences have been collec ted
                if len(memory) > self.mini_batch_size:
                    # Sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain num of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y) vs episodes (X)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        
        plt.subplot(1, 2, 1)

        plt.xlabel('Episodes')
        plt.ylabel('Mean rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y) vs episodes (X)
        plt.subplot(1, 2, 2)
        plt.xlabel('Time steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

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

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)[0])
            else:
                # Calculate target q values (expected return)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # Calculate q values from current policy network
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)
    
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
    