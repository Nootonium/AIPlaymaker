from typing import List, Optional


class PlayerColor:
    WHITE = "white"
    BLACK = "black"


class MoveType:
    COPY = "COPY"
    JUMP = "JUMP"
    INVALID = "INVALID"


BoardState = List[List[Optional[str]]]


class Position:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col


class GameBoard:
    def __init__(self, new_board: BoardState, current_turn: str):
        self.board = new_board
        self.current_turn = current_turn

    def is_move_valid(self, start: Position, target: Position) -> str:
        row_diff = abs(target.row - start.row)
        col_diff = abs(target.col - start.col)

        if self.board[start.row][start.col] != self.current_turn:
            return MoveType.INVALID

        if self.board[target.row][target.col] is not None:
            return MoveType.INVALID

        if row_diff <= 1 and col_diff <= 1:
            return MoveType.COPY

        if (row_diff == 2 and col_diff <= 2) or (col_diff == 2 and row_diff <= 2):
            return MoveType.JUMP

        return MoveType.INVALID

    def get_adjacent_pieces(self, position: Position) -> List[Position]:
        directions = [-1, 0, 1]
        adjacent_positions = [
            Position(position.row + d_row, position.col + d_col)
            for d_row in directions
            for d_col in directions
            if d_row != 0 or d_col != 0
        ]

        return [
            pos
            for pos in adjacent_positions
            if 0 <= pos.row < len(self.board) and 0 <= pos.col < len(self.board[0])
        ]

    def make_move(self, start: Position, target: Position) -> bool:
        move_type = self.is_move_valid(start, target)

        if move_type == MoveType.INVALID:
            print("Invalid move!")
            return False

        self.board[target.row][target.col] = self.current_turn

        if move_type == MoveType.JUMP:
            self.board[start.row][start.col] = None

        self.convert_opponent_pieces(target)
        self.switch_turn()
        return True

    def convert_opponent_pieces(self, position: Position) -> None:
        adjacent_pieces = self.get_adjacent_pieces(position)
        opponent = (
            PlayerColor.WHITE
            if self.current_turn == PlayerColor.BLACK
            else PlayerColor.BLACK
        )

        for cell in adjacent_pieces:
            if self.board[cell.row][cell.col] == opponent:
                self.board[cell.row][cell.col] = self.current_turn

    def switch_turn(self) -> None:
        self.current_turn = (
            PlayerColor.BLACK
            if self.current_turn == PlayerColor.WHITE
            else PlayerColor.WHITE
        )

    def is_game_complete(self) -> bool:
        no_empty_cells = all(cell is not None for row in self.board for cell in row)

        scores = self.get_scores()
        no_pieces_left = scores["white"] == 0 or scores["black"] == 0

        return no_empty_cells or no_pieces_left

    def get_scores(self) -> dict:
        white_score = sum(
            1 for row in self.board for cell in row if cell == PlayerColor.WHITE
        )
        black_score = sum(
            1 for row in self.board for cell in row if cell == PlayerColor.BLACK
        )
        return {"white": white_score, "black": black_score}

    def get_winner(self) -> str:
        scores = self.get_scores()
        if scores["white"] == 0:
            return PlayerColor.BLACK
        elif scores["black"] == 0:
            return PlayerColor.WHITE

        if scores["white"] > scores["black"]:
            return PlayerColor.WHITE
        elif scores["white"] < scores["black"]:
            return PlayerColor.BLACK
        else:
            return "draw"

    def get_next_possible_moves(self) -> List[dict]:
        possible_moves = []
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                if self.board[row][col] == self.current_turn:
                    start_pos = Position(row, col)
                    # Check all possible moves within the range for COPY and JUMP
                    for d_row in [-2, -1, 0, 1, 2]:
                        for d_col in [-2, -1, 0, 1, 2]:
                            if d_row == 0 and d_col == 0:
                                continue
                            target_row = row + d_row
                            target_col = col + d_col
                            if 0 <= target_row < len(
                                self.board
                            ) and 0 <= target_col < len(self.board[0]):
                                target_pos = Position(target_row, target_col)
                                move_type = self.is_move_valid(start_pos, target_pos)
                                if move_type != MoveType.INVALID:
                                    possible_moves.append(
                                        {
                                            "start": start_pos,
                                            "target": target_pos,
                                            "move_type": move_type,
                                        }
                                    )
        return possible_moves


if __name__ == "__main__":
    initial_board = [
        [None, None, None, None, None],
        [None, "white", "white", "white", None],
        [None, "black", "black", "black", None],
        [None, None, None, None, None],
        [None, None, None, None, None],
    ]
    game = GameBoard(initial_board, PlayerColor.WHITE)
    game.make_move(Position(1, 1), Position(2, 1))
