from collections import Counter
from typing import Literal, Tuple
from .c4_constants import VALID_MOVES


DIRECTIONS = [
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
]


class C4Board:
    state: str
    dimensions: Tuple[int, int]

    def __init__(self, dimensions: Tuple[int, int], state: str) -> None:
        self.dimensions = dimensions
        self.state = state
        if not self.is_valid_game_state():
            raise ValueError("Invalid board")

    def __str__(self) -> str:
        rows, columns = self.dimensions
        lines = []
        for row in range(rows):
            line = []
            for i in range(row * columns, (row + 1) * columns):
                if self.state[i] == "1":
                    line.append("\033[31mO\033[0m")  # Red color for 'X'
                elif self.state[i] == "2":
                    line.append("\033[34mO\033[0m")  # Blue color for 'O'
                else:
                    line.append("-")
            lines.append(" | ".join(line))
        lines.append("-" * ((4 * columns) - 1))  # Add a bottom line to the board
        return "\n".join(reversed(lines))

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

    def with_move(self, column: int) -> "C4Board":
        # returns a new board with the move made
        rows, columns = self.dimensions
        next_player = self.get_next_player()
        if not (0 <= column < columns):
            raise ValueError(f"Column {column} is out of range")

        new_state = self.state
        for i in range(rows):
            if new_state[i * columns + column] == " ":
                new_state = (
                    new_state[: i * columns + column]
                    + next_player
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

    def find_move_position(self, new_board: str) -> Tuple[int, int]:
        rows, columns = self.dimensions
        diff_positions = [
            i for i in range(rows * columns) if self.state[i] != new_board[i]
        ]
        if len(diff_positions) > 1:
            raise ValueError(
                "Invalid board state: more than one move was made in a single turn"
            )
        if len(diff_positions) == 0:
            raise ValueError("Invalid board state: no move was made")

        position = diff_positions.pop()
        return position // columns, position % columns

    def _count_consecutive_pieces(self, game_state, start_position, player, direction):
        rows, columns = self.dimensions
        row, col = start_position
        count = 0
        dx, dy = direction

        while 0 <= row < rows and 0 <= col < columns and game_state[row][col] == player:
            count += 1
            row += dx
            col += dy

        return count

    def get_winner(self) -> str | None:
        if len(self.get_next_possible_moves()) == 0:
            return " "
        rows, columns = self.dimensions
        state = self.state
        game_state = [state[i : i + columns] for i in range(0, len(state), columns)]

        for r in range(rows):
            for c in range(columns):
                if game_state[r][c] != " ":
                    for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if (
                            self._count_consecutive_pieces(
                                game_state, (r, c), game_state[r][c], direction
                            )
                            >= 4
                        ):
                            return game_state[r][c]

        if " " in state:
            return None
        return " "

    def blocks_opponent_win(self, position: Tuple[int, int], player: str) -> bool:
        rows, columns = self.dimensions
        opponent = "2" if player == "1" else "1"

        for dx, dy in DIRECTIONS:
            count = 0
            x, y = position[0] + dx, position[1] + dy
            while 0 <= x < rows and 0 <= y < columns:
                cell = self.state[x * columns + y]
                if cell == " " or cell == player:
                    break
                elif cell == opponent:
                    count += 1
                x, y = x + dx, y + dy
            if count >= 3:
                return True

        return False
