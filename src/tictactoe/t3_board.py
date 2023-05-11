from collections import Counter
from enum import Enum
from typing import Any, List, Literal, Union, Type


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
    ]  # type: List[List[int]]

    def __init__(self, input_board) -> None:
        self.input_format = self.detect_format(input_board)
        self.state = self.convert_to_internal_format(input_board, self.input_format)

    @staticmethod
    def detect_format(input_board: str | list) -> Formats:
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
    def convert_to_internal_format(
        input_board: Union[str, List[Any]], board_format: Formats
    ) -> str:
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                return "".join("".join(row) for row in input_board)
            case T3Board.Formats.FLAT_LIST:
                return "".join(input_board)
            case T3Board.Formats.STRING:
                return str(input_board)
            case _:
                return None

    def is_valid_game_state(self) -> bool:
        return self.validate_state(self.state)

    @classmethod
    def validate_state(cls: Type["T3Board"], state: str) -> bool:
        return all(move in cls.VALID_MOVES for move in state)

    def game_over(self) -> Literal["X", "O", " "] | None:
        return self.determine_winner(self.state)

    @classmethod
    def determine_winner(
        cls: Type["T3Board"], state: str
    ) -> Literal["X", "O", " "] | None:
        # implementation here
        game_state = list(state)
        for wins in cls.WINNING:
            # Create a tuple
            win = (game_state[wins[0]], game_state[wins[1]], game_state[wins[2]])
            if win == ("X", "X", "X"):
                return "X"
            if win == ("O", "O", "O"):
                return "O"
        # Check for stalemate
        if " " in game_state:
            return None
        return " "

    def get_next_possible_moves(self) -> list[int]:
        return self.calculate_possible_moves(self.state)

    @classmethod
    def calculate_possible_moves(cls: Type["T3Board"], state: str) -> list[int]:
        game_state = list(state)
        return [i for i, p in enumerate(game_state) if p == " "]

    def get_next_player(self) -> Literal["X", "O"]:
        return self.calculate_next_player(self.state)

    @classmethod
    def calculate_next_player(cls: Type["T3Board"], state: str) -> Literal["X", "O"]:
        count = Counter(state)
        return "X" if count.get("X", 0) <= count.get("O", 0) else "O"

    @staticmethod
    def validate_board(input_board: str) -> bool:
        board_format = T3Board.detect_format(input_board)
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
                            or cell not in T3Board.VALID_MOVES
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
                        or cell not in T3Board.VALID_MOVES
                    ):
                        return False
                return True

            case T3Board.Formats.STRING:
                if not isinstance(input_board, str) or len(input_board) != 9:
                    return False
                for cell in input_board:
                    if cell not in T3Board.VALID_MOVES:
                        return False
                return True

            case _:
                return False

    def get_format(self) -> Formats:
        return self.input_format

    def convert_to_output_format(self) -> str | list[list[str]] | list[str]:
        return self.convert_from_internal_format(self.state, self.input_format)

    @staticmethod
    def convert_from_internal_format(
        state: str, board_format: Formats
    ) -> str | list[list[str]] | list[str]:
        match board_format:
            case T3Board.Formats.NESTED_LIST:
                return [[state[i * 3 + j] for j in range(3)] for i in range(3)]
            case T3Board.Formats.FLAT_LIST:
                return list(state)
            case T3Board.Formats.STRING:
                return state
            case _:
                raise ValueError("Invalid board format")

    def is_empty(self) -> bool:
        return self.state == " " * 9
