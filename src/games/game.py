# This is a boilerplate file for a new game. It should be copied and renamed to the new game's name.
# Every method in this file should be overridden in the new game's file.
#
# The game file should be placed in the src/games directory.
#
import random
from typing import Dict, List, Tuple


class Game:
    def __init__(self):
        self.state_space: int = 0
        self.action_space: int = 0

        self.reset()

    def reset(self) -> List[int]:
        """Reset the game state and return the initial information state.

        This method is used to reset the game state at the beginning of each episode. It should
        clean up any state information and return the initial information state for the game.

        Returns:
            List[int]: The initial information state.
        """
        raise NotImplementedError("The reset method must be overridden")

    def step(self, action: int) -> Tuple[List[int], int, bool, Dict]:
        """Take a step in the game by performing the given action.

        This method is used to advance the game state by performing the given action. It should
        return the next information state, the reward for the action (for the current player who
        performed the action), a flag indicating whether the game is done (terminal), and any additional
        information in a dictionary. We do not make use of the additional information in NFSP but
        if you are making any modifications you may need to use it.

        Args:
            action (int): The action to perform.

        Returns:
            Tuple[List[int], int, bool, Dict]: The next information state, the reward for the action,
            a flag indicating whether the game is done, and any additional information.
        """
        raise NotImplementedError("The step method must be overridden")
