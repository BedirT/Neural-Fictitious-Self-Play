from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.memory import RLMemory, SLMemory
from networks.as_net import ASNet
from networks.br_net import BRNet


class NFSPAgent:
    """Implements the NFSP agent, combining both RL and SL strategies."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        anticipatory_param: float,
        rl_learning_rate: float = 1e-4,
        sl_learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        """Initializes the NFSP agent with both a best-response (BR-Net) and average-strategy (AS-Net) network.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
            anticipatory_param (float): Anticipatory parameter to balance between BR and AS
                strategies.
            rl_learning_rate (float, optional): Learning rate for the best-response network.
                Defaults to 1e-4.
            sl_learning_rate (float, optional): Learning rate for the average-strategy network.
                Defaults to 1e-4.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            device (str, optional): Device to run the agent on. Defaults to "cpu".
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.anticipatory_param = anticipatory_param
        self.device = device
        self.gamma = gamma

        self.last_action_best_response = (
            False  # Whether the last action was best-response based
        )

        self.rl_memory = RLMemory(capacity=10000)  # RL experiences memory
        self.sl_memory = SLMemory(capacity=10000)  # SL experiences memory

        self.br_net = BRNet(state_dim, action_dim).to(device)  # Best-response network
        self.as_net = ASNet(state_dim, action_dim).to(
            device
        )  # Average-strategy network
        self.target_br_net = BRNet(state_dim, action_dim).to(
            device
        )  # Target best-response network
        self.update_target_network()

        self.br_optimizer = optim.Adam(self.br_net.parameters(), lr=rl_learning_rate)
        self.as_optimizer = optim.Adam(self.as_net.parameters(), lr=sl_learning_rate)

    def select_action(self, state: List[int], epsilon: float = 0.1) -> int:
        """Selects an action using an epsilon-greedy strategy for the RL part and a deterministic
        strategy for the SL part based on the anticipatory parameter.

        Args:

            epsilon (float): The exploration rate for epsilon-greedy action selection.

        Returns:
            int: Selected action.
        """
        # Convert state to tensor for network input
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # With probability (1 - anticipatory_param), use the average strategy network
        if np.random.rand() > self.anticipatory_param:
            self.last_action_best_response = False
            with torch.no_grad():
                probabilities = self.as_net(state_tensor)
                action = probabilities.multinomial(
                    num_samples=1
                ).item()  # Sampled action based
                # on probabilities
        else:
            # Epsilon-greedy action selection for exploring RL network's strategy
            self.last_action_best_response = True
            if np.random.rand() < epsilon:
                action = np.random.choice(range(self.action_dim))  # Random action
            else:
                # Use the best-response network with the highest probability
                with torch.no_grad():
                    q_values = self.br_net(state_tensor)
                    action = q_values.max(1)[
                        1
                    ].item()  # Action with the highest Q-value

        return action

    def store_experience_rl(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        """Stores an experience tuple in the RL memory.

        Args:
            state (List[int]): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (List[int]): The next state after the action.
            done (bool): Whether the episode has ended.
        """
        self.rl_memory.add_experience(state, action, reward, next_state, done)

    def store_experience_sl(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        """Stores an experience tuple in the SL memory.

        Args:
            state (List[int]): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (List[int]): The next state after the action.
            done (bool): Whether the episode has ended.
        """
        self.sl_memory.add_experience(state, action, reward, next_state, done)

    def update_target_network(self) -> None:
        """Updates the target best-response network with the weights of the current best-response
        network.
        """
        self.target_br_net.load_state_dict(self.br_net.state_dict())

    def train(self, batch_size: int) -> None:
        """Trains both the Best-Response Network (BR-Net) and the Average-Strategy Network (AS-Net)

        Trains both the Best-Response Network (BR-Net) and the Average-Strategy Network (AS-Net)
        using sampled experiences from their respective memories.

        Args:
            batch_size (int): Batch size for training.
        """
        if len(self.rl_memory) >= batch_size:
            # Sample from RL memory for BR-Net training
            rl_states, rl_actions, rl_rewards, rl_next_states, rl_dones = zip(
                *self.rl_memory.sample(batch_size)
            )

            # Convert to tensors
            rl_states = torch.FloatTensor(rl_states).to(self.device)
            rl_next_states = torch.FloatTensor(rl_next_states).to(self.device)
            rl_actions = torch.LongTensor(rl_actions).view(-1, 1).to(self.device)
            rl_rewards = torch.FloatTensor(rl_rewards).to(self.device)
            rl_dones = torch.FloatTensor(rl_dones).to(self.device)

            # BR-Net Q-learning update
            # Using the formula: Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a))
            q_values = self.br_net(rl_states).gather(1, rl_actions)  # Q(s, a)
            next_q_values = (
                self.target_br_net(rl_next_states).max(1)[0].detach()
            )  # max(Q(s', a'))
            expected_q_values = rl_rewards + self.gamma * next_q_values * (
                1 - rl_dones
            )  # r + gamma * max(Q(s', a')) * (1 - done)
            # Note: 1 - done is used to zero out the Q-value if the next state is terminal
            # Note: detach() is used to prevent backpropagation through next_q_values (target network)
            # Note: gather() is used to select the Q-values for the actions taken
            # Note: There is a subtle difference between the formula and the code. Can you spot it?
            #       Yes, the formula is using - Q(s, a) while the code does not. This is because
            #       the loss function already includes a minus sign, so we are effectively minimizing
            #       the negative Q-value, which is equivalent to maximizing the Q-value.

            # Compute loss and backpropagate for BR-Net
            br_loss = F.mse_loss(q_values.squeeze(-1), expected_q_values)
            self.br_optimizer.zero_grad()
            br_loss.backward()
            self.br_optimizer.step()

        if len(self.sl_memory) >= batch_size:
            # Sample from SL memory for AS-Net training
            sl_states, sl_actions, _, _, _ = self.sl_memory.sample(batch_size)

            # Convert to tensors
            sl_states = torch.FloatTensor(sl_states).to(self.device)
            sl_actions = torch.LongTensor(sl_actions).to(self.device)

            # AS-Net supervised learning update
            action_probs = self.as_net(sl_states)
            as_loss = F.cross_entropy(action_probs, sl_actions)

            # Compute loss and backpropagate for AS-Net
            self.as_optimizer.zero_grad()
            as_loss.backward()
            self.as_optimizer.step()
