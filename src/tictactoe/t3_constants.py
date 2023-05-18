from enum import Enum


class BoardFormats(Enum):
    FLAT_LIST = 1
    STRING = 2
    INVALID = 3


VALID_MOVES = ["X", "O", " "]

WINNING = [
    [0, 1, 2],  # Across top
    [3, 4, 5],  # Across middle
    [6, 7, 8],  # Across bottom
    [0, 3, 6],  # Down left
    [1, 4, 7],  # Down middle
    [2, 5, 8],  # Down right
    [0, 4, 8],  # Diagonal ltr
    [2, 4, 6],  # Diagonal rtl
]
