from collections import Counter
from typing import Literal, Set, Tuple
from .c4_constants import VALID_MOVES


class C4Board:
    state: str
    dimensions: Tuple[int, int]

    def __init__(self, dimensions: Tuple[int, int], state: str) -> None:
        self.dimensions = dimensions
        self.state = state
        if not self.is_valid_game_state():
            raise ValueError("Invalid board")

    def has_floating_piece(self) -> bool:
        rows, columns = self.dimensions
        grid = [list(self.state[i * columns : (i + 1) * columns]) for i in range(rows)]
        for col in range(columns):
            has_piece = False
            for row in reversed(range(rows)):
                if grid[row][col] != " ":
                    has_piece = True
                elif has_piece:
                    return True
        return False

    def is_valid_game_state(self) -> bool:
        rows, columns = self.dimensions
        return (
            len(self.state) == rows * columns
            and all(move in VALID_MOVES for move in self.state)
            and not self.has_floating_piece()
        )

    def get_next_possible_moves(self) -> list[int]:
        possible_moves = []
        rows, columns = self.dimensions
        for col in range(columns):
            for i in reversed(range(rows)):
                if self.state[i * columns + col] == " ":
                    possible_moves.append(col)
                    break
        return possible_moves

    def with_move(self, column: int, player: Literal["1", "2"]) -> "C4Board":
        # returns a new board with the move made
        rows, columns = self.dimensions
        if not (0 <= column < columns):
            raise ValueError(f"Column {column} is out of range")

        new_state = self.state
        for i in range(rows):
            if new_state[i * columns + column] == " ":
                new_state = (
                    new_state[: i * columns + column]
                    + player
                    + new_state[i * columns + column + 1 :]
                )
                return C4Board(self.dimensions, new_state)

        raise ValueError(f"Column {column} is full.")

    def get_next_player(self) -> Literal["1", "2"]:
        count = Counter(self.state)
        return "1" if count.get("1", 0) <= count.get("2", 0) else "2"

    def is_empty(self) -> bool:
        rows, columns = self.dimensions
        return self.state == (" " * rows * columns)

    def find_move_position(self, new_board: str) -> int | None:
        rows, columns = self.dimensions
        diff_positions = [
            i for i in range(rows * columns) if self.state[i] != new_board[i]
        ]
        if len(diff_positions) > 1:
            raise ValueError(
                "Invalid board state: more than one move was made in a single turn"
            )
        if diff_positions:
            return diff_positions.pop() % columns
        return None

    def get_winner(self) -> str | None:
        rows, columns = self.dimensions
        state = self.state
        game_state = [state[i : i + columns] for i in range(0, len(state), columns)]

        # depth first search to count the number of consecutive pieces
        # couldve used a for loop but this is more fun
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
                        visited: Set = set()
                        if dfs((r, c), game_state[r][c], direction, visited) >= 4:
                            return game_state[r][c]

        if " " in state:
            return None
        return " "
