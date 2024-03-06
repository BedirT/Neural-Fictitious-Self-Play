import random

BET = 1
PASS = 0


class NashEquilibriumAgent:
    def __init__(self, position: int, action_space: int = None):
        self.position = position  # 0 for the first player, 1 for the second player

    def select_action(self, state):
        card = state[0]
        history = state[1:4]
        turn = state[4]
        bets = state[5:]
        alpha = random.uniform(
            0, 1 / 3
        )  # α in [0, 1/3] for the first player's mixed strategy
        opp_action = history[turn - 1] if turn > 0 else None

        # First player strategy
        if self.position == 0:
            if turn == 0:  # First action
                if card == 1:  # Jack
                    return BET if random.random() < alpha else PASS
                elif card == 2:  # Queen
                    return PASS  # Always checks
                else:  # King
                    return BET if random.random() < 3 * alpha else PASS
            else:  # Second action
                if card == 1:
                    return PASS  # Always folds/checks
                elif card == 2:  # Queen
                    if opp_action == PASS:
                        return PASS  # Always checks if opponent checks
                    else:  # Opponent bets
                        return BET if random.random() < alpha + (1 / 3) else PASS
                else:  # King
                    if opp_action == PASS:
                        return BET if random.random() < 3 * alpha else PASS
                    else:
                        return BET  # Always bets if opponent bets

        # Second player strategy
        else:
            # Second player only has one action
            if card == 1:  # Jack
                if opp_action == PASS:
                    return BET if random.random() < 1 / 3 else PASS
                else:
                    return PASS
            elif card == 2:  # Queen
                if opp_action == PASS:
                    return PASS
                else:
                    return BET if random.random() < 1 / 3 else PASS
            else:  # King
                return BET  # Always bets/calls
