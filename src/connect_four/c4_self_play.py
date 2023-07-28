import random
from tqdm import tqdm
import numpy as np
import torch
from torch import load

from .c4_board import C4Board
from .c4_mcts import C4MCTreeSearch
from .c4_converter import encode_board
from .c4_net import Connect4Net


class Player:
    def __init__(self):
        self.wins = 0

    def make_move(self, board: C4Board):
        print(board)
        moves = board.get_next_possible_moves()
        move = int(input(f"Enter move {moves}: "))
        return board.with_move(move)

    def reset_wins(self):
        self.wins = 0

    def won(self):
        self.wins += 1


class MCTSPlayer(Player):
    def __init__(self, num_iterations=1000, c_param=1.4):
        super(MCTSPlayer, self).__init__()  # corrected here
        self.num_iterations = num_iterations
        self.c_param = c_param

    def make_move(self, board: C4Board):
        mcts = C4MCTreeSearch(board, self.c_param)
        new_board = mcts.run(self.num_iterations)
        return new_board


class NeuralNetPlayer(Player):
    def __init__(self, in_model):
        super(NeuralNetPlayer, self).__init__()
        self.model = in_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_move(self, board):
        board_state = encode_board(board)
        board_state_tensor = (
            torch.tensor(board_state, dtype=torch.float)
            .unsqueeze(0)
            .to(device=self.device)
        )
        with torch.inference_mode():
            q_values = self.model(board_state_tensor)
        move = torch.argmax(q_values).item()
        if move not in board.get_next_possible_moves():
            print("Model is trying to make an invalid move.")
            print(f"Board: {board}")
            print(f"Q values: {q_values}")
            print(f"Move: {move}")
            print(f"Possible moves: {board.get_next_possible_moves()}")
            move = random.choice(board.get_next_possible_moves())
        return board.with_move(move)


class RandomPlayer(Player):
    def __init__(self):
        super(RandomPlayer, self).__init__()

    def make_move(self, board):
        move = random.choice(board.get_next_possible_moves())
        return board.with_move(move)


def play_game(player1: Player, player2: Player):
    rows, columns = (6, 7)
    board = C4Board((rows, columns), " " * (rows * columns))

    while board.get_winner() is None:
        board = player1.make_move(board)
        if board.get_winner() is not None:
            break
        board = player2.make_move(board)
    print(board)
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


def tune_simulations_hyperparameters():
    num_simulations = np.arange(4000, 50000, 1000)
    for s in num_simulations:
        print(f"Trying {s} simulations")
        p1 = MCTSPlayer(s)
        p2 = MCTSPlayer(s + 500)
        scores = play_games(p1, p2, 100)
        if scores[0] > scores[1]:
            print(f"Player 1 wins with {s} simulations")
            break


def tune_c_param():
    c_params = np.arange(1.2, 1.6, 0.1)
    best_c = 0.9
    for c in c_params:
        print(f"Trying {c} for c")
        mcts1 = MCTSPlayer(3500, c)
        mcts2 = MCTSPlayer(3500, best_c)
        scores = play_games(mcts1, mcts2, 100)
        if scores[0] > scores[1]:
            print(f"Player 1 wins with {c} for c")
            best_c = c
    print(f"Best c: {best_c}")


if __name__ == "__main__":
    p1 = MCTSPlayer(3500, 1.4)
    model = Connect4Net(7)
    model.load_state_dict(load("connect_four/models/first.pth"))
    model.eval()
    p2 = NeuralNetPlayer(model)
    play_games(p1, p2, 2)
