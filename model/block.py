import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        identity = x
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x += identity
        x = self.relu(x)
        return x