import json
from ..exceptions import (
    InvalidBoardException,
    GameFinishedException,
    UnsupportedDimensionsException,
)
from .c4_board import C4Board
from .c4_mcts import C4MCTreeSearch
from .c4_converter import C4Converter


class C4Service:
    def get_board(self, input_board, dimensions):
        is_valid, board_format = C4Converter.validate_board(input_board, dimensions)
        if not is_valid:
            raise InvalidBoardException()
        board = C4Board(
            dimensions,
            C4Converter.convert_to_internal_format(input_board, board_format),
        )
        return board, board_format

    def build_response(self, move, post_move_board, board_format) -> dict:
        post_move_board = C4Converter.convert_from_internal_format(
            post_move_board, board_format
        )
        return {"move": move, "post_move_board": post_move_board}

    def get_next_move(self, input_board, dimensions) -> dict:
        board, board_format = self.get_board(input_board, dimensions)

        if board.get_winner() is not None:
            raise GameFinishedException()

        mcts = C4MCTreeSearch(board)
        res = mcts.run(500)

        _, col = board.find_move_position(res.state)
        return self.build_response(col, res.state, board_format)
