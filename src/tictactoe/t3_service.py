"""Service module for TicTacToe game."""
import random
from .t3_board import T3Board
from .t3_tree import T3Tree

from ..exceptions import InvalidBoardException


class T3Service:
    def validate_board(self, input_board) -> None:
        is_valid = T3Board.validate_board(input_board)
        if not is_valid:
            raise InvalidBoardException()

    def create_board(self, input_board) -> T3Board:
        self.validate_board(input_board)
        return T3Board(input_board)

    def create_tree(self, board: T3Board) -> T3Tree:
        return T3Tree(board)

    def get_next_move(self, input_board) -> dict:
        board = self.create_board(input_board)
        if board.is_empty():
            return self.random_move_on_empty_board()

        tree = self.create_tree(board)
        best_move = tree.get_best_next_move()
        return best_move

    def get_next_moves(self, input_board) -> list:
        board = self.create_board(input_board)
        tree = self.create_tree(board)
        best_moves = tree.get_best_next_moves()
        return best_moves

    def convert_to_output_format(self, board: T3Board):
        return board.convert_to_output_format()

    def random_move_on_empty_board(self) -> dict:
        choice = random.choice(range(9))
        new_board = T3Board("" * 9)
        post_move_board = new_board.get_next_possible_moves()[choice]
        return {"move": choice, "post_move_board": post_move_board}
