import random
from typing import Dict, List, Tuple


class KuhnPoker:
    def __init__(self):
        """Initialize the Kuhn Poker game."""
        self.cards = [1, 2, 3]  # 1 is Jack, 2 is Queen, 3 is King
        self.ante = 1  # initial bet

        self.max_state_len = 3
        self.state_space = (
            self.max_state_len + 4
        )  # players card + history len + players bet + opponents bet
        self.action_space = 2  # pass or bet

        self.reset()

    def reset(self) -> List[int]:
        """Reset the game state and return the initial information state.

        Returns:
            List[int]: The initial information state.
        """
        self.deck = self.cards.copy()
        random.shuffle(self.deck)
        self.current_player = 0
        self.history = ""
        self.done = False
        self.winner = None
        self.state = [self.deck[0], self.deck[1], self.ante, self.ante]
        # (player0 card, player1 card, player0 bet, player1 bet)
        return self.get_info_state()

    def history_to_list(self) -> List[int]:
        """Convert the history string to a list of action integers.

        Returns:
            List[int]: The list of action integers.
        """
        history_list = [0] * self.max_state_len
        for i, action in enumerate(self.history):
            history_list[i] = self.string_to_action_int(action)
        return history_list

    def get_info_state(self) -> List[int]:
        """Get the current information state.

        Returns:
            List[int]: The current information state.
        """
        # Players Card, Action History, Turn, Player0 Bet, Player1 Bet
        return [
            self.state[self.current_player],
            *self.history_to_list(),
            len(self.history),
            self.state[2],
            self.state[3],
        ]

    def action_int_to_string(self, action: int) -> str:
        """Convert the action integer to a string representation.

        Args:
            action (int): The action integer.

        Returns:
            str: The string representation of the action.
        """
        return "p" if action == 0 else "b"

    def string_to_action_int(self, action: str) -> int:
        """Convert the action string to an integer representation.

        Args:
            action (str): The action string.

        Returns:
            int: The integer representation of the action.
        """
        return 0 if action == "p" else 1

    def step(self, action: int) -> Tuple[List[int], List[int], bool, Dict]:
        """Take a step in the game by performing the given action.

        Args:
            action (int): The action to perform.

        Returns:
            Tuple[List[int], List[int], bool, Dict]: The next information state, rewards, done flag, and additional info.
        """
        assert not self.done, "Game is over"
        str_action = self.action_int_to_string(action)

        self.history += str_action

        # Betting Logic
        if str_action == "b":
            self.state[self.current_player + 2] += 1

        # Terminal state
        if self.history in ["bb", "pbb", "bp", "pp", "pbp"]:
            self.done = True
            self.determine_winner()
        else:
            self.current_player = 1 - self.current_player

        return (
            self.get_info_state(),
            self.get_reward(self.current_player),
            self.done,
            {},
        )

    def determine_winner(self):
        """Determine the winner of the game based on the history."""
        if self.history in ["bb", "pbb", "pp"]:
            self.winner = 0 if self.state[0] > self.state[1] else 1
        elif self.history == "bp":
            self.winner = 0
        elif self.history == "pbp":
            self.winner = 1
        else:
            raise ValueError(f"Invalid history: {self.history}")

    def get_reward(self, player: int = -1) -> List[int]:
        """Get the reward for the specified player.

        Args:
            player (int, optional): The player index. If -1, return the reward for both players. Defaults to -1.

        Returns:
            List[int]: The reward(s).
        """
        if not self.done:
            res = [0, 0]
        res = (
            [-self.state[2], self.state[3]]
            if self.winner == 1
            else [self.state[2], -self.state[3]]
        )
        return res if player == -1 else res[player]
