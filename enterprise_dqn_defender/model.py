import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.network(x)