# flappy-bird-dql

This is a PyTorch implementation of a Deep Q-Network (DQN) that learns to play **Flappy Bird** from the [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) environment.
This implementation includes optional **Double DQN** and **Dueling DQN** extensions and a configurable training pipeline that also works on classic control tasks like CartPole.

## Demos
| Flappy Bird | CartPole |
|-------------|----------|
| ![Flappy Bird](demos/flappybird_demo.mov) | ![CartPole](demos/cartpole2_demo.mov) |

Trained agents and reward curves are saved under `runs/`.

## Features
- **Deep Q-Network** with experience replay.
- **Double DQN** which reduces Q-value overestimation by separating action selection from evaluation. Toggle `enable_double_dqn` in hyperparameters.
- **Dueling DQN** which separates value and advantage streams to prevent wasted time in training. Toggle `enable_dueling_dqn`.
- **Epsilon-greedy** exploration with a configurable decay.
- **Hyperparameters** configurable per each experiment.
- **Automatic logging**, **best-model checkpointing**, and **live reward/epsilon plots**.

## Project Layout
| File | Purpose |
|------|---------|
| `agent.py` | Training/inference loop, optimization, logging, and plotting |
| `dqn.py` | The `DQN` network (standard and dueling architectures) |
| `experience_replay.py` | `ReplayMemory` buffer for experience replay |
| `hyperparameters.yml` | Named hyperparameter sets (cartpole1–3, flappybird1–2) |
| `runs/` | Per-experiment `.log`, `.pt` (weights), and `.png` training curves |
| `demos/` | Recorded gameplay videos! |

## Setup
Note: requires **Python 3.11** 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train an agent using a named hyperparameter set from `hyperparameters.yml`:

```bash
python agent.py HYPERPARAM_SET_NAME --train
```

Watch a trained agent play (loads `runs/<set>.pt`, renders to screen):

```bash
python agent.py HYPERPARAM_SET_NAME
```

While training, the script writes:
- `runs/<set>.log`. The timestamped log of new best rewards.
- `runs/<set>.pt`. The best model weights so far.
- `runs/<set>.png`. Curves of mean-reward and epsilon decay.

## Play yourself!

Play as a human.
```bash
flappy_bird_gymnasium
```

Play against a random agent.
```bash
flappy_bird_gymnasium --mode random
```

## Hyperparameter sets

Defined in `hyperparameters.yml`. Add a new key to define your own experiment.

Notable hyperparameters:
| Key | Meaning |
|-----|---------|
| `env_id` | Gymnasium environment (e.g. `FlappyBird-v0`, `CartPole-v1`) |
| `replay_memory_size` | Experience replay buffer capacity |
| `mini_batch_size` | Samples per optimization step |
| `epsilon_init` / `_decay` / `_min` | Exploration schedule |
| `network_sync_rate` | Steps between target-network syncs |
| `learning_rate_a` | Adam learning rate |
| `discount_factor_g` | Reward discount (gamma) |
| `fc1_nodes` | Hidden layer width |
| `enable_double_dqn` | Use Double DQN target computation |
| `enable_dueling_dqn` | Use dueling value/advantage architecture |

