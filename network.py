"""
    Simple Neural Network with 8 layers

    Returns:
        _type_: _description_
"""

import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(70, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1024)
        self.fc6 = nn.Linear(1024, 256)
        self.fc7 = nn.Linear(256, 16)
        self.fc8 = nn.Linear(16, 1)
        self.activation = nn.LeakyReLU()  # Consistent activation function
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, training=True):
        x = self.activation(self.fc1(x))
        if training:
            x = self.dropout(x)

        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        if training:
            x = self.dropout(x)

        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        x = self.activation(self.fc7(x))
        if training:
            x = self.dropout(x)

        x = self.fc8(x)
        return x
