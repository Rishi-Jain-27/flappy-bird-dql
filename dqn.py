import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        # state_dim is the dim of input layer
        # action dim is the dim of the output layer
        super(DQN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn
        self.fc1 = nn.Linear(in_features=state_dim,
                             out_features=hidden_dim)
        
        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(in_features=hidden_dim, out_features=256)
            self.value = nn.Linear(256, 1) # Get value down to 1

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
            
        else:
            self.fc2 = nn.Linear(in_features=hidden_dim,
                                    out_features=action_dim)
        
    
    def forward(self, x):
        # x is the state (the 12 informational values)
        
        x = F.relu(self.fc1(x))
        
        if self.enable_dueling_dqn:
            # Value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calc
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calc Q
            return (V + A - torch.mean(A, dim=1, keepdim=True))
        else:
            return self.output(x)


if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim=state_dim,
              action_dim=action_dim)
    
    # 1x12 because PyTorch uses the first dim for batching
    state = torch.randn(1, state_dim)

    output = net(state)
    print(output)
