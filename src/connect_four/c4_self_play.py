from tqdm import tqdm
from .c4_board import C4Board
from .c4_mcts import C4MCTreeSearch
import numpy as np


class Player:
    def __init__(self):
        self.wins = 0

    def make_move(self, board: C4Board):
        print(board)
        moves = board.get_next_possible_moves()
        move = int(input(f"Enter move {moves}: "))
        return board.with_move(move, board.get_next_player())

    def reset_wins(self):
        self.wins = 0

    def won(self):
        self.wins += 1


class MCTSPlayer(Player):
    def __init__(self, num_iterations=1000):
        super(MCTSPlayer, self).__init__()  # corrected here
        self.num_iterations = num_iterations

    def make_move(self, board: C4Board):
        mcts = C4MCTreeSearch(board)
        new_board = mcts.run(self.num_iterations)
        return new_board


class NeuralNetPlayer(Player):
    def __init__(self, model):
        self.model = model

    def make_move(self, board):
        pass


def play_game(player1: Player, player2: Player):
    rows, columns = (6, 7)
    board = C4Board((rows, columns), " " * (rows * columns))

    while board.get_winner() is None:
        board = player1.make_move(board)
        if board.get_winner() is not None:
            break
        board = player2.make_move(board)

    res = board.get_winner()
    if res == "1":
        player1.won()
    if res == "2":
        player2.won()
    return


def play_games(player1, player2, num_games=100):
    for i in tqdm(range(num_games)):
        if i % 2 == 0:
            play_game(player1, player2)
        else:
            play_game(player2, player1)

    p1_wins = player1.wins
    p2_wins = player2.wins
    draws = num_games - (p1_wins + p2_wins)
    player1.reset_wins()
    player2.reset_wins()
    scores = [p1_wins, p2_wins, draws]
    print(f"Player 1 wins: {p1_wins}, Player 2 wins: {p2_wins}, Draws: {draws}")
    return scores


def generate_training_data():
    pass


def tune_hyperparameters():
    num_simulations = np.arange(4000, 50000, 1000)
    for s in num_simulations:
        print(f"Trying {s} simulations")
        p1 = MCTSPlayer(s)
        p2 = MCTSPlayer(s + 500)
        scores = play_games(p1, p2, 100)
        if scores[0] > scores[1]:
            print(f"Player 1 wins with {s} simulations")
            break


if __name__ == "__main__":
    tune_hyperparameters()
