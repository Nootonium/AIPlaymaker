from .t3Board import T3Board
from .t3Tree import T3Tree


class Service:
    def _create_tree_from_board(self, input_board):
        board_format = T3Board.detect_input_board_format(input_board)
        t3tree_board = T3Board.convert_input_board_to_t3tree_format(
            input_board, board_format
        )
        return T3Tree(t3tree_board), board_format

    def get_next_move(self, input_board):
        tree, board_format = self._create_tree_from_board(input_board)
        best_move = tree.get_best_next_move()
        output_board = T3Board.convert_t3tree_format_to_output_board(
            best_move["board"], board_format
        )
        return {"move": best_move["move"], "board": output_board}

    def get_next_moves(self, input_board):
        tree, board_format = self._create_tree_from_board(input_board)
        best_moves = tree.get_best_next_moves()
        output_boards = [
            T3Board.convert_t3tree_format_to_output_board(move["board"], board_format)
            for move in best_moves
        ]
        return output_boards
