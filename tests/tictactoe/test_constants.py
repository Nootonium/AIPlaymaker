# Constants for detect_format tests

VALID_NESTED_LIST = [["X  "], ["O  "], ["   "]]
INVALID_NESTED_LIST_1 = [
    ["X  "],
    ["O  "],
    "   ",
]
INVALID_NESTED_LIST_2 = [["X  "], ["O  "]]  # Nested list does not have 3 items

VALID_FLAT_LIST = ["X", " ", " ", "O", " ", " ", " ", " ", " "]
INVALID_FLAT_LIST_1 = [
    "X",
    "X",
    "X",
    "O",
    "O",
    "O",
    " ",
    " ",
    " ",
    "X",
]
INVALID_FLAT_LIST_2 = [
    "X",
    "X",
    "X",
    "O",
    "O",
    "O",
    " ",
]

VALID_STRING = "X  O     "
INVALID_STRING_1 = "XXXOOO    "
INVALID_STRING_2 = "XXXOOO"

INVALID_STRING_STATE = "XXXOFO   "

INVALID_INPUT_1 = 123  # Input is not a string or list
INVALID_INPUT_2 = {"X": "X", "O": "O", " ": " "}  # Input is not a string or list
