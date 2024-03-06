import random
from typing import Dict, List, Tuple


class Card:
    rank_values = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }

    def __init__(self, rank, suit):
        if suit not in ["H", "D", "S", "C"]:
            raise ValueError("Invalid suit")
        if rank not in Card.rank_values.keys():
            raise ValueError("Invalid rank")
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __lt__(self, other):
        return Card.rank_values[self.rank] < Card.rank_values[other.rank]

    def __hash__(self):
        return hash((self.rank, self.suit))


class Deck:
    def __init__(self):
        self.cards = [
            Card(rank, suit)
            for suit in ["H", "D", "S", "C"]
            for rank in Card.rank_values
        ]

    def __str__(self):
        return f"Deck of {len(self.cards)} cards"

    def shuffle(self):
        if len(self) < 52:
            raise ValueError("Only full decks can be shuffled")
        random.shuffle(self.cards)

    def deal(self, num_cards=1):
        if len(self) < num_cards:
            raise ValueError("Not enough cards in deck to deal")
        return [self.cards.pop() for _ in range(num_cards)]

    def is_empty(self):
        return len(self.cards) == 0

    def __add__(self, other):
        if isinstance(other, Deck):
            combined_deck = Deck()
            combined_deck.cards = self.cards + other.cards
            return combined_deck
        else:
            raise ValueError("Only decks can be added together")

    def __len__(self):
        return len(self.cards)


class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def receive_cards(self, cards):
        self.hand.extend(cards)

    def play_cards(self, cards_to_play, claim, number):
        if len(cards_to_play) != number:
            return [], False
        for card in cards_to_play:
            if card not in self.hand:
                return [], False
        for card in cards_to_play:
            self.hand.remove(card)
        return cards_to_play, True

    def show_hand(self):
        return ", ".join(str(card) for card in self.hand)


class CheatGame:
    raise NotImplementedError("Cheat class not implemented yet")
