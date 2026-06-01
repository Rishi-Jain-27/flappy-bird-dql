import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        # state_dim is the dim of input layer
        # action dim is the dim of the output layer
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=state_dim,
                             out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim,
                             out_features=action_dim)
    
    def forward(self, x):
        # x is the state (the 12 informational values)
        # send state through layer one, and then through ReLU
        # send that to the output layer
        x = F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim=state_dim,
              action_dim=action_dim)
    
    # 1x12 because PyTorch uses the first dim for batching
    state = torch.randn(1, state_dim)

    output = net(state)
    print(output)
