from ..exceptions import InvalidBoardException, GameFinishedException
from .c4_board import C4Board
from .c4_mcts import C4Node, C4MCTS
from .c4_converter import C4Converter


class C4Service:
    def get_next_move(self, input_board, dimensions=(6, 7)) -> dict:
        is_valid, board_format = C4Converter.validate_board(input_board, dimensions)

        if not is_valid:
            raise InvalidBoardException()

        board = C4Board(
            dimensions,
            C4Converter.convert_to_internal_format(input_board, board_format),
        )
        if board.get_winner() is not None:
            raise GameFinishedException()

        root = C4Node(board)
        mcts = C4MCTS(root)

        res = mcts.run(1000)
        move = board.find_move_position(res)
        if res:
            post_move_board = C4Converter.convert_from_internal_format(
                res, board_format
            )

        return {"move": move, "post_move_board": post_move_board}
