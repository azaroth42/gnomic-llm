"""Cards and decks. Ace=1, Jack=11, Queen=12, King=13. Suits are unordered."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

SUITS = ("S", "H", "D", "C")
SUIT_NAMES = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs"}
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
RED_SUITS = frozenset({"H", "D"})
RANK_LABELS = {1: "A", 11: "J", 12: "Q", 13: "K"}
LABEL_RANKS = {"A": 1, "J": 11, "Q": 12, "K": 13}
_WORD_RANKS = {
    "ace": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "jack": 11, "queen": 12, "king": 13,
}
_SYMBOL_SUITS = {v: k for k, v in SUIT_SYMBOLS.items()}


@dataclass(frozen=True, order=True)
class Card:
    rank: int  # 1..13
    suit: str  # one of SUITS

    def __post_init__(self) -> None:
        if not 1 <= self.rank <= 13 or self.suit not in SUITS:
            raise ValueError(f"invalid card rank={self.rank} suit={self.suit}")

    @property
    def label(self) -> str:
        return RANK_LABELS.get(self.rank, str(self.rank))

    @property
    def is_red(self) -> bool:
        return self.suit in RED_SUITS

    @property
    def color(self) -> str:
        return "red" if self.is_red else "black"

    @property
    def is_face(self) -> bool:
        return self.rank >= 11

    @property
    def is_male(self) -> bool:
        """Jacks and Kings are male."""
        return self.rank in (11, 13)

    @property
    def is_female(self) -> bool:
        """Queens are female."""
        return self.rank == 12

    @property
    def gender(self) -> str | None:
        return "male" if self.is_male else "female" if self.is_female else None

    @property
    def is_even(self) -> bool:
        return self.rank % 2 == 0

    def __str__(self) -> str:
        return f"{self.label}{self.suit}"

    def pretty(self) -> str:
        return f"{self.label}{SUIT_SYMBOLS[self.suit]}"

    def to_json(self) -> str:
        return str(self)

    @classmethod
    def parse(cls, text: str) -> "Card":
        """Parse '10H', 'qs', 'A♠', '10 of hearts', 'queen of spades'."""
        s = text.strip()
        m = re.fullmatch(r"(?i)\s*(10|[2-9]|[AJQK])\s*([SHDC♠♥♦♣])\s*", s)
        if m:
            r, su = m.group(1).upper(), m.group(2)
            rank = LABEL_RANKS.get(r) or int(r)
            suit = _SYMBOL_SUITS.get(su, su.upper())
            return cls(rank, suit)
        m = re.fullmatch(r"(?i)\s*(\w+)\s+of\s+(spades|hearts|diamonds|clubs)\s*", s)
        if m:
            word = m.group(1).lower()
            rank = _WORD_RANKS.get(word) or (int(word) if word.isdigit() else 0)
            suit = m.group(2)[0].upper()
            return cls(rank, suit)
        raise ValueError(f"cannot parse card: {text!r}")


def full_deck() -> list[Card]:
    return [Card(r, s) for s in SUITS for r in range(1, 14)]


def shuffled_deck(rng: random.Random) -> list[Card]:
    deck = full_deck()
    rng.shuffle(deck)
    return deck
