from collections import Counter
from typing import Literal, Type
from .c4_constants import VALID_MOVES


class Board:
    def __init__(self, dimensions: tuple, state: str) -> None:
        self.dimensions = dimensions
        self.state = state
        if not self.is_valid_game_state():
            raise ValueError("Invalid board")

    def is_valid_game_state(self) -> bool:
        return self.validate_state(self.state, self.dimensions)

    @classmethod
    def validate_state(cls: Type["Board"], state: str, dimension: tuple) -> bool:
        rows, colums = dimension
        return (
            all(move in VALID_MOVES for move in state) and len(state) == rows * colums
        )

    def get_next_possible_moves(self) -> list[int]:
        return self.calculate_possible_moves(self.state)

    @classmethod
    def calculate_possible_moves(cls: Type["Board"], state: str) -> list[int]:
        game_state = list(state)
        return [i for i, p in enumerate(game_state) if p == " "]

    def get_next_player(self) -> Literal["1", "2"]:
        return self.calculate_next_player(self.state)

    @classmethod
    def calculate_next_player(cls: Type["Board"], state: str) -> Literal["1", "2"]:
        count = Counter(state)
        return "1" if count.get("1", 0) <= count.get("2", 0) else "2"

    def is_empty(self) -> bool:
        rows, columns = self.dimensions
        return self.state == " " * rows * columns

    def find_move_position(self, new_board: str) -> int | None:
        return self.compare_board_states(self.state, new_board, self.dimensions)

    @classmethod
    def compare_board_states(
        cls: Type["Board"], old_board: str, new_board: str, dimensions: tuple
    ) -> int | None:
        rows, columns = dimensions
        diff_positions = [
            i for i in range(rows * columns) if old_board[i] != new_board[i]
        ]

        if len(diff_positions) > 1:
            raise ValueError(
                "Invalid board state: more than one move was made in a single turn"
            )
        elif diff_positions:
            return diff_positions.pop()
        else:
            return None

    def get_winner(self) -> str | None:
        rows, columns = self.dimensions
        state = self.state
        game_state = [state[i : i + columns] for i in range(0, len(state), columns)]

        def dfs(position, player, direction, visited):
            row, col = position
            if (
                not (0 <= row < rows)
                or not (0 <= col < columns)
                or game_state[row][col] != player
                or (row, col) in visited
            ):
                return 0
            visited.add(position)
            dx, dy = direction
            return 1 + dfs((row + dx, col + dy), player, direction, visited)

        for r in range(rows):
            for c in range(columns):
                if game_state[r][c] != " ":
                    for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        visited = set()
                        if dfs((r, c), game_state[r][c], direction, visited) >= 4:
                            return game_state[r][c]

        if " " in state:
            return None
        return " "
