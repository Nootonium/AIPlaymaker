from typing import Type
from .t3_board import T3Board
from .t3_tree import T3Tree

from ..exceptions import InvalidBoardException


class Service:
    def _create_tree_from_board(self, input_board) -> Type["T3Tree"]:
        self.validate_board(input_board)
        board = T3Board(input_board)
        return T3Tree(board)

    def get_next_move(self, input_board) -> dict:
        tree = self._create_tree_from_board(input_board)
        best_move = tree.get_best_next_move()
        return {"move": best_move["move"], "board": best_move["board"]}

    def get_next_moves(self, input_board) -> dict:
        tree = self._create_tree_from_board(input_board)

        best_moves = tree.get_best_next_moves()
        print(best_moves)
        return {}

    def validate_board(self, input_board):
        is_valid = T3Board.validate_board(input_board)
        if not is_valid:
            raise InvalidBoardException()
        return is_valid

    def convert_to_output_format(self, board: T3Board):
        return board.convert_to_output_format()
