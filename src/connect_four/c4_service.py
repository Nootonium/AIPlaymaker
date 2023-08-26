import json
from torch import load
from ..exceptions import (
    InvalidBoardException,
    GameFinishedException,
    UnsupportedDimensionsException,
)
from .c4_board import C4Board
from .c4_mcts import C4MCTreeSearch
from .c4_converter import C4Converter
from .c4_nets import Connect4Net
from .c4_self_play import NeuralNetPlayer


class C4Service:
    def __init__(self) -> None:
        with open(
            "src/connect_four/models/model1_config.json", "r", encoding="utf-8"
        ) as file:
            configs = json.load(file)
        conv_config = configs["conv_config"]
        fc_config = configs["fc_config"]
        model = Connect4Net(conv_config, fc_config)
        model.load_state_dict(load("src/connect_four/models/model1_epoch_7.pth"))
        self.nn_player = NeuralNetPlayer(model)

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

    def get_next_move(self, input_board, dimensions, strategy) -> dict:
        if strategy == "ml":
            if dimensions != (6, 7):
                raise UnsupportedDimensionsException(
                    "ML strategy only supports 6x7 board"
                )
            strategy_fn = self._ml_strategy
        else:
            strategy_fn = self._mcts_strategy

        return self._get_move_based_on_strategy(input_board, dimensions, strategy_fn)

    def _get_move_based_on_strategy(self, input_board, dimensions, strategy_fn) -> dict:
        board, board_format = self.get_board(input_board, dimensions)

        if board.get_winner() is not None:
            raise GameFinishedException()

        res = strategy_fn(board)

        _, col = board.find_move_position(res.state)
        return self.build_response(col, res.state, board_format)

    def _ml_strategy(self, board):
        return self.nn_player.make_move(board)

    def _mcts_strategy(self, board):
        mcts = C4MCTreeSearch(board)
        return mcts.run(500)
