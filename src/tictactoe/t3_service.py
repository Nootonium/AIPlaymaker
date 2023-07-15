"""Service module for TicTacToe game."""
import random

from .t3_board import T3Board
from .t3_tree import T3Tree
from .t3_converter import T3Converter
from .t3_constants import BoardFormats
from .t3_net import T3Net, load_model, predict_move

from ..exceptions import InvalidBoardException, InvalidActionException


class T3Service:
    def __init__(self) -> None:
        model = T3Net()
        self.model = load_model(model, "src/resources/models/model_216_0.0003_23.pth")

    def get_board(self, input_board):
        is_valid, board_format = T3Converter.validate_board(input_board)
        if not is_valid:
            raise InvalidBoardException()
        board = T3Board(
            T3Converter.convert_to_internal_format(input_board, board_format)
        )
        return board, board_format

    def build_response(self, move, post_move_board, board_format) -> dict:
        post_move_board = T3Converter.convert_from_internal_format(
            post_move_board, board_format
        )
        return {"move": move, "post_move_board": post_move_board}

    def get_next_move(self, input_board, strategy) -> dict:
        if strategy == "ml":
            return self.get_next_ml_move(input_board)
        else:
            return self.get_next_algo_move(input_board)

    def get_next_moves(self, input_board, strategy) -> list:
        if strategy == "ml":
            raise InvalidActionException("ML strategy does not support multiple moves")
        else:
            return self.get_next_algo_moves(input_board)

    def get_next_algo_move(self, input_board) -> dict:
        board, board_format = self.get_board(input_board)

        if board.is_empty():
            return self.random_move_on_empty_board(board_format)

        tree = T3Tree(board)
        best_move = tree.get_best_next_move()

        if best_move is None:
            raise Exception("No best move found")

        move, post_move_board = best_move

        return self.build_response(move, post_move_board, board_format)

    def get_next_algo_moves(self, input_board) -> list:
        board, board_format = self.get_board(input_board)
        response = []
        if board.is_empty():
            response = self.all_possible_moves_on_empty_board(board_format)
        else:
            tree = T3Tree(board)
            best_moves = tree.get_best_next_moves()
            for move, post_move_board in best_moves:
                response.append(
                    self.build_response(move, post_move_board, board_format)
                )

        return response

    def random_move_on_empty_board(self, board_format) -> dict:
        choice = random.choice(range(9))
        empty_board = T3Board("" * 9)
        new_board = empty_board.make_move(choice)
        return self.build_response(choice, new_board.state, board_format)

    def all_possible_moves_on_empty_board(self, board_format) -> list:
        empty_board = T3Board(" " * 9)

        ans = []
        for move in range(9):
            new_board = empty_board.make_move(move)
            ans.append(self.build_response(move, new_board.state, board_format))
        return ans

    def convert_back_to_external_format(
        self, move: dict, board_format: BoardFormats
    ) -> dict:
        if "post_move_board" in move:
            move["post_move_board"] = T3Converter.convert_from_internal_format(
                move["post_move_board"], board_format
            )
        return move

    def get_next_ml_move(self, input_board) -> dict:
        board, board_format = self.get_board(input_board)

        move = predict_move(self.model, board)
        new_board = board.make_move(move)

        return {"move": move, "post_move_board": new_board}
