from collections import Counter
from typing import Literal
from .t3_constants import WINNING, VALID_MOVES
from ..exceptions import InvalidBoardException


class T3Board:
    def __init__(self, input_board: str):
        self.state = input_board
        if not self.is_valid_game_state():
            raise InvalidBoardException("Invalid board state")

    def __str__(self):
        pretty_board = ""
        for i in range(0, 9, 3):
            pretty_board += "| " + " | ".join(self.state[i : i + 3]) + " |\n"
        return pretty_board

    def is_valid_game_state(self) -> bool:
        return all(move in VALID_MOVES for move in self.state) and len(self.state) == 9

    def get_winner(self) -> Literal["X", "O", " "] | None:
        game_state = list(self.state)
        for wins in WINNING:
            win = (game_state[wins[0]], game_state[wins[1]], game_state[wins[2]])
            if win == ("X", "X", "X"):
                return "X"
            if win == ("O", "O", "O"):
                return "O"

        if " " in game_state:
            return None
        return " "

    def get_next_possible_moves(self) -> list[int]:
        game_state = list(self.state)
        return [i for i, p in enumerate(game_state) if p == " "]

    def get_next_player(self) -> Literal["X", "O"]:
        count = Counter(self.state)
        return "X" if count.get("X", 0) <= count.get("O", 0) else "O"

    def is_empty(self) -> bool:
        return self.state == " " * 9

    def find_move_position(self, new_board: str) -> int | None:
        diff_positions = [i for i in range(9) if self.state[i] != new_board[i]]

        if len(diff_positions) > 1:
            raise ValueError(
                "Invalid board state: more than one move was made in a single turn"
            )
        if diff_positions:
            return diff_positions[0]
        return None

    def make_move(self, move: int) -> "T3Board":
        if move not in self.get_next_possible_moves():
            print(self.state)
            print(move)
            move = self.get_next_possible_moves()[0]  # TODO: temp fix for ml model
        next_player = self.get_next_player()
        new_board_state = list(self.state)
        new_board_state[move] = next_player
        return T3Board("".join(new_board_state))
