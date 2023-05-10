from enum import Enum


class T3Board:
    class Formats(Enum):
        NESTED_LIST = 1
        FLAT_LIST = 2
        STRING = 3

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

    def __init__(self, input_board):
        self.input_format = self.detect_format(input_board)
        self.state = self.convert_to_internal_format(input_board, self.input_format)

    @staticmethod
    def detect_format(input_board):
        if isinstance(input_board, list) and all(
            isinstance(row, list) and len(row) == 1 for row in input_board
        ):
            return T3Board.Formats.NESTED_LIST

        if isinstance(input_board, list) and len(input_board) == 9:
            return T3Board.Formats.FLAT_LIST

        if isinstance(input_board, str) and len(input_board) == 9:
            return T3Board.Formats.STRING

        raise ValueError("Invalid input board format")

    @staticmethod
    def convert_to_internal_format(input_board, board_format):
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                return "".join("".join(row) for row in input_board)
            case T3Board.Formats.FLAT_LIST:
                return "".join(input_board)
            case T3Board.Formats.STRING:
                return input_board
            case _:
                return None

    def is_valid_game_state(self):
        return self.validate_state(self.state)

    @classmethod
    def validate_state(cls, state):
        return all(move in cls.VALID_MOVES for move in state)

    def game_over(self):
        return self.determine_winner(self.state)

    @classmethod
    def determine_winner(cls, state):
        # implementation here
        game_state = list(state)
        for wins in cls.WINNING:
            # Create a tuple
            w = (game_state[wins[0]], game_state[wins[1]], game_state[wins[2]])
            if w == ("X", "X", "X"):
                return "X"
            if w == ("O", "O", "O"):
                return "O"
        # Check for stalemate
        if " " in game_state:
            return None
        return " "

    def get_next_possible_moves(self):
        return self.calculate_possible_moves(self.state)

    @classmethod
    def calculate_possible_moves(cls, state):
        game_state = list(state)
        return [i for i, p in enumerate(game_state) if p == " "]

    @classmethod
    def isvalidate_state(cls, input_board):
        board_format = cls.detect_format(input_board)
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                if not isinstance(input_board, list) or len(input_board) != 3:
                    return False
                for row in input_board:
                    if not isinstance(row, list) or len(row) != 3:
                        return False
                    for cell in row:
                        if (
                            not isinstance(cell, str)
                            or len(cell) != 1
                            or cell not in cls.VALID_MOVES
                        ):
                            return False
                return True

            case T3Board.Formats.FLAT_LIST:
                if not isinstance(input_board, list) or len(input_board) != 9:
                    return False
                for cell in input_board:
                    if (
                        not isinstance(cell, str)
                        or len(cell) != 1
                        or cell not in cls.VALID_MOVES
                    ):
                        return False
                return True

            case T3Board.Formats.STRING:
                if not isinstance(input_board, str) or len(input_board) != 9:
                    return False
                for cell in input_board:
                    if cell not in cls.VALID_MOVES:
                        return False
                return True

            case _:
                return False

    @staticmethod
    def convert_from_internal_format(state, board_format):
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                return [[state[i * 3 + j] for j in range(3)] for i in range(3)]
            case T3Board.Formats.FLAT_LIST:
                return list(state)
            case T3Board.Formats.STRING:
                return state
            case _:
                raise ValueError("Invalid board format")
