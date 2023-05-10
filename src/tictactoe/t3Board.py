from enum import Enum


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

    def convert_input_board_to_t3tree_format(input_board, board_format):
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                return "".join("".join(row) for row in input_board)
            case T3Board.Formats.FLAT_LIST:
                return "".join(input_board)
            case T3Board.Formats.STRING:
                return input_board
            case _:
                return None

    def validate_board(input_board, board_format):
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
                            or cell not in T3Board.MOVES
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
                        or cell not in T3Board.MOVES
                    ):
                        return False
                return True

            case T3Board.Formats.STRING:
                if not isinstance(input_board, str) or len(input_board) != 9:
                    return False
                for cell in input_board:
                    if cell not in T3Board.MOVES:
                        return False
                return True

            case _:
                return False

    def convert_t3tree_format_to_output_board(t3tree_board, board_format):
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                return [[t3tree_board[i * 3 + j] for j in range(3)] for i in range(3)]
            case T3Board.Formats.FLAT_LIST:
                return list(t3tree_board)
            case T3Board.Formats.STRING:
                return t3tree_board
            case _:
                raise ValueError("Invalid board format")
