from .c4_board import C4Board
from .c4_mcts import C4MCTS, C4Node


class SelfPlay:
    def __init__(self, board_size=(6, 7), num_games=100):
        self.board_size = board_size
        self.num_games = num_games

    def play_one_game(self):
        board = C4Board(self.board_size, " " * self.board_size[0] * self.board_size[1])
        node = C4Node(board)
        mcts = C4MCTS(node)

        while node.board.get_winner() is None:
            move = mcts.run(1000)
            node = C4Node(C4Board(self.board_size, move))
            yield (board, move)

    def generate_training_data(self):
        training_data = []
        for _ in range(self.num_games):
            for board, move in self.play_one_game():
                training_data.append((board, move))
        return training_data
