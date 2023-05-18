"""Service module for TicTacToe game."""
import random

from .t3_board import T3Board
from .t3_tree import T3Tree
from .t3_converter import T3Converter
from .t3_constants import BoardFormats

from ..exceptions import InvalidBoardException


class T3Service:
    def get_next_move(self, input_board) -> dict:
        is_valid, board_format = T3Converter.validate_board(input_board)

        if not is_valid:
            raise InvalidBoardException()

        board = T3Board(
            T3Converter.convert_to_internal_format(input_board, board_format)
        )
        if board.is_empty():
            return self.random_move_on_empty_board()

        tree = T3Tree(board)
        best_move = tree.get_best_next_move()

        if best_move:
            self.convert_back_to_external_format(best_move, board_format)

        return best_move

    def get_next_moves(self, input_board) -> list:
        is_valid, board_format = T3Converter.validate_board(input_board)

        if not is_valid:
            raise InvalidBoardException()

        board = T3Board(
            T3Converter.convert_to_internal_format(input_board, board_format)
        )
        response = []
        if board.is_empty():
            response = self.all_possible_moves_on_empty_board()
        else:
            tree = T3Tree(board)
            response = tree.get_best_next_moves()

        for move in response:
            self.convert_back_to_external_format(move, board_format)

        return response

    def random_move_on_empty_board(self) -> dict:
        choice = random.choice(range(9))
        new_board = T3Board("" * 9)
        post_move_board = new_board.get_next_possible_moves()[choice]
        return {"move": choice, "post_move_board": post_move_board}

    def all_possible_moves_on_empty_board(self) -> list:
        empty_board = T3Board(" " * 9)

        ans = []
        for move_index in empty_board.get_next_possible_moves():
            new_board = (
                empty_board.state[:move_index]
                + "X"
                + empty_board.state[move_index + 1 :]
            )
            ans.append({"move": move_index, "post_move_board": new_board})
        return ans

    def convert_back_to_external_format(
        self, move: dict, board_format: BoardFormats
    ) -> dict:
        if "post_move_board" in move:
            move["post_move_board"] = T3Converter.convert_from_internal_format(
                move["post_move_board"], board_format
            )
        return move
