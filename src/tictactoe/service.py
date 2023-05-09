from tictactoe.t3Board import T3Board
from tictactoe.t3Tree import T3Tree


class Service:
    def get_next_move(self, input_board):
        tree = T3Tree(T3Board.convert_input_board_to_t3tree_format(input_board, format))
        best_move = tree.get_best_next_move()
        updated_board = board.apply_move(best_move)
        return {"move": best_move, "board": updated_board.format()}

    def get_next_moves(self, input_board):
        board = T3Board(input_board)
        tree = T3Tree(board)
        best_moves = tree.get_best_next_moves()
        return [
            {"move": move, "board": board.apply_move(move).format()}
            for move in best_moves
        ]
