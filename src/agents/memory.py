# agents/memory.py

import random
from collections import deque
from typing import Any, List, Tuple

from utils.reservoir_sampling import ReservoirSampling


class RLMemory:
    """Memory for storing and sampling reinforcement learning experiences."""

    def __init__(self, capacity: int) -> None:
        """Initializes the RL memory with a given capacity.

        Args:
            capacity (int): The maximum number of experiences to hold.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add_experience(
        self, state: Any, action: Any, reward: Any, next_state: Any, done: bool
    ) -> None:
        """Stores an experience tuple in the memory.

        Args:
            state (Any): The current state.
            action (Any): The action taken.
            reward (Any): The reward received.
            next_state (Any): The next state after the action.
            done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple[Any, Any, Any, Any, bool]]:
        """Samples a batch of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            List[Tuple[Any, Any, Any, Any, bool]]: A batch of sampled experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the current size of the memory.

        Returns:
            int: The current size of the memory.
        """
        return len(self.memory)


class SLMemory:
    """Memory for storing and sampling supervised learning experiences using reservoir sampling."""

    def __init__(self, capacity: int) -> None:
        """Initializes the SL memory with a given capacity.

        Args:
            capacity (int): The maximum number of experiences to hold.
        """
        self.reservoir_sampling = ReservoirSampling(capacity)

    def add_experience(
        self, state: Any, action: Any, reward: Any, next_state: Any, done: bool
    ) -> None:
        """Adds an experience to the reservoir sampling.

        Args:
            state (Any): The current state.
            action (Any): The action taken.
            reward (Any): The reward received.
            next_state (Any): The next state after the action.
            done (bool): Whether the episode has ended.
        """
        self.reservoir_sampling.add((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[bool]]:
        """Samples a batch of experiences from the reservoir.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple[List[Any], List[Any], List[Any], List[Any], List[bool]]: A batch of sampled experiences.
        """
        sampled_experiences = random.sample(
            self.reservoir_sampling.get_sample(), batch_size
        )
        states, actions, rewards, next_states, dones = zip(*sampled_experiences)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Returns the current size of the reservoir.

        Returns:
            int: The current size of the reservoir.
        """
        return len(self.reservoir_sampling)
