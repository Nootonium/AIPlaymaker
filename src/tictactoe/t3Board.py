from enum import Enum
from exceptions import InvalidBoardFormatError


class T3Board:
    class Formats(Enum):
        NESTED_LIST = 1
        FLAT_LIST = 2
        STRING = 3

    MOVES = ["X", "O", " "]

    def detect_input_board_format(input_board):
        if isinstance(input_board, list) and all(
            isinstance(row, list) and len(row) == 1 for row in input_board
        ):
            return T3Board.Formats.NESTED_LIST

        if isinstance(input_board, list) and len(input_board) == 9:
            return T3Board.Formats.FLAT_LIST

        if isinstance(input_board, str) and len(input_board) == 9:
            return T3Board.Formats.STRING

        raise ValueError("Invalid input board format")

    def convert_input_board_to_t3tree_format(self, input_board, board_format):
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                return "".join("".join(row) for row in input_board)
            case T3Board.Formats.FLAT_LIST:
                return "".join(input_board)
            case T3Board.Formats.STRING:
                return input_board
            case _:
                raise ValueError("Invalid board format")

    def validate_board(self, input_board, board_format):
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                # Check if input_board is a list of 3 lists, each with 3 strings of length 1
                if not isinstance(input_board, list) or len(input_board) != 3:
                    return False
                for row in input_board:
                    if not isinstance(row, list) or len(row) != 3:
                        return False
                    for cell in row:
                        if (
                            not isinstance(cell, str)
                            or len(cell) != 1
                            or cell not in self.MOVES
                        ):
                            return False
                return True

            case T3Board.Formats.FLAT_LIST:
                # Check if input_board is a list of 9 strings of length 1
                if not isinstance(input_board, list) or len(input_board) != 9:
                    return False
                for cell in input_board:
                    if (
                        not isinstance(cell, str)
                        or len(cell) != 1
                        or cell not in self.MOVES
                    ):
                        return False
                return True

            case T3Board.Formats.STRING:
                # Check if input_board is a string of length 9
                if not isinstance(input_board, str) or len(input_board) != 9:
                    return False
                for cell in input_board:
                    if cell not in self.MOVES:
                        return False
                return True

            case _:
                return False
