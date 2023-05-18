from collections import Counter
from typing import Literal, Type
from .t3_constants import WINNING, VALID_MOVES


class T3Board:
    def __init__(self, input_board: str) -> None:
        self.state = input_board
        if not self.is_valid_game_state():
            raise ValueError("Invalid board")

    def is_valid_game_state(self) -> bool:
        return self.validate_state(self.state)

    @classmethod
    def validate_state(cls: Type["T3Board"], state: str) -> bool:
        return all(move in VALID_MOVES for move in state) and len(state) == 9

    def get_winner(self) -> Literal["X", "O", " "] | None:
        return self.determine_winner(self.state)

    @classmethod
    def determine_winner(
        cls: Type["T3Board"], state: str
    ) -> Literal["X", "O", " "] | None:
        # implementation here
        game_state = list(state)
        for wins in WINNING:
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

    def is_empty(self) -> bool:
        return self.state == " " * 9

    def find_move_position(self, new_board: str) -> int | None:
        return self.compare_board_states(self.state, new_board)

    @classmethod
    def compare_board_states(
        cls: Type["T3Board"], old_board: str, new_board: str
    ) -> int | None:
        diff_positions = [i for i in range(9) if old_board[i] != new_board[i]]

        if len(diff_positions) > 1:
            raise ValueError(
                "Invalid board state: more than one move was made in a single turn"
            )
        elif diff_positions:
            return diff_positions[0]
        else:
            return None
