import torch
import torch.nn as nn
import torch.nn.functional as F


class ASNet(nn.Module):
    """Defines the neural network architecture for the average-strategy network."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        Initializes the AS-Net with a simple feedforward architecture.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
        """
        super(ASNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, action_dim)  # Output layer

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Action probabilities.
        """
        x = F.relu(self.fc1(state))  # Activation function for first layer
        x = F.relu(self.fc2(x))  # Activation function for second layer
        return F.softmax(self.fc3(x), dim=-1)  # Softmax to get probabilities
