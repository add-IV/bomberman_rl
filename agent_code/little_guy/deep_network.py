"""
This module contains the model for the evaluation and policy network of the MCTS.
Input is the local game state in a pattern around the agent.
The pattern is in a diamond shape with the agent in the center, the width of the diamond is 1 + 2 * 4 = 9.
We will remove the spaces that are in the middle of the sides of the diamond.
So one layer of the diamond has 9 + 2 * 7 + 2 * 3 + 4 * 1 - 1 = 36 fields.
This is a perfect square, so we can use a 6x6 stack of layers. The convolutions make sense since neighbor relations will be conserved.
Multiple layers exist for coins, crates, bombs, explosions, walls, and agents.
So the input is a 6x6x6 stack.
The first Block is a convolutional Block.
Then there is a number of residual Blocks.
Lastly, there are two heads, one for the value and one for the policy.
The value head is a single neuron with a tanh activation function.
It outputs the value of the state in the range of -1 to 1.
The policy head outputs the probabilities for each action.
"""

import torch
import torch.nn as nn

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]


class ConvolutionalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=2**7, kernel_size=3, stride=1, padding=1
    ):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=2**7, kernel_size=3, stride=1, padding=1
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += residual
        x = self.relu(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, in_channels, out_size):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(12, out_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.tanh(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, in_channels, out_size=6):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2, 1)
        self.batch_norm = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(24, out_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class MCTSNetwork(nn.Module):
    def __init__(self, in_channels=7, hidden_size=2**7, residual_blocks=1):
        super().__init__()
        self.conv_block = ConvolutionalBlock(in_channels)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(residual_blocks)]
        )
        self.value_head = ValueHead(hidden_size, 1)
        self.policy_head = PolicyHead(hidden_size)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy


def load_model(path):
    model = MCTSNetwork()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
