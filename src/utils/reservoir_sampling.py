import random


class ReservoirSampling:
    """Implements reservoir sampling for efficient sampling from SL memory."""

    def __init__(self, capacity: int) -> None:
        """Initializes the reservoir with a given capacity.

        Args:
            capacity: The maximum number of items that the reservoir can hold.
        """
        self.capacity = capacity
        self.items = []
        self.index = 0

    def add(self, item: object) -> None:
        """Adds an item to the reservoir.

        If the reservoir is not full, the item is added directly. Otherwise, it replaces an
        existing item with a probability that ensures uniform sampling.

        Args:
            item: The item to add to the reservoir.
        """
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:  # ! STILL HAZY ON THIS
            s = int(random.random() * self.index)
            if s < self.capacity:
                self.items[s] = item
        self.index += 1

    def get_sample(self):
        """Returns the items in the reservoir."""
        return self.items

    def __len__(self):
        return len(self.items)
