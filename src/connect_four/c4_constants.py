from enum import Enum


class BoardFormats(Enum):
    STRING = 1
    FLAT_LIST = 2
    INVALID = -1


VALID_MOVES = [" ", "1", "2"]
