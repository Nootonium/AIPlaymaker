import random
from tqdm import tqdm
import torch
import numpy as np

from .c4_board import C4Board
from .c4_mcts import C4MCTreeSearch
from .c4_converter import encode_board


class Player:
    def __init__(self):
        self.wins = 0

    def make_move(self, board: C4Board):
        print(board)
        moves = board.get_next_possible_moves()
        move = int(input(f"Enter move {moves}: "))
        return board.with_move(move)

    def make_move_with_q_values(self, board: C4Board):
        print(board)
        moves = board.get_next_possible_moves()
        move = int(input(f"Enter move {moves}: "))
        return board.with_move(move), move, None

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

    def make_move_with_q_values(self, board: C4Board):
        mcts = C4MCTreeSearch(board, self.c_param)
        new_board = mcts.run(self.num_iterations)
        q_values_dict = mcts.root.get_q_values()

        q_values_list = []
        for col in range(board.dimensions[1]):
            q_values_list.append(q_values_dict.get(col, float("-inf")))

        _, col = board.find_move_position(new_board.state)
        return new_board, col, np.array(q_values_list).reshape(1, -1)


class NeuralNetPlayer(Player):
    def __init__(self, in_model):
        super(NeuralNetPlayer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = in_model.to(self.device)
        self.model.eval()

    def make_move(self, board):
        board_state = encode_board(board)
        board_state_tensor = (
            torch.tensor(board_state, dtype=torch.float)
            .unsqueeze(0)
            .to(device=self.device)
        )
        with torch.inference_mode():
            q_values = self.model(board_state_tensor)

        # Get the sorted Q-values and corresponding moves
        _, sorted_moves = torch.topk(q_values, k=q_values.size(-1))

        for move in sorted_moves[0]:
            if move.item() in board.get_next_possible_moves():
                return board.with_move(move.item())

        raise Exception("No valid moves found")

    def make_move_with_q_values(self, board, epsilon=0.01):
        board_state = encode_board(board)
        board_state_tensor = (
            torch.tensor(board_state, dtype=torch.float)
            .unsqueeze(0)
            .to(device=self.device)
        )

        with torch.inference_mode():
            q_values_tensor = self.model(board_state_tensor)
            q_values = q_values_tensor.cpu().numpy()

        if random.random() < epsilon:
            random_move = random.choice(board.get_next_possible_moves())
            return (board.with_move(random_move), random_move, q_values)

        sorted_moves = np.argsort(q_values[0])[::-1]

        for move in sorted_moves:
            if move.item() in board.get_next_possible_moves():
                return (board.with_move(move.item()), move.item(), q_values)

        raise Exception("No valid moves found")


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
    res = board.get_winner()
    if res == "1":
        player1.won()
    if res == "2":
        player2.won()
    return


def play_games(player1, player2, num_games=100, verbose=False):
    iterable = tqdm(range(num_games)) if verbose else range(num_games)

    for i in iterable:
        players = (player1, player2) if i % 2 == 0 else (player2, player1)
        play_game(*players)

    draws = num_games - (player1.wins + player2.wins)

    scores = [player1.wins, player2.wins, draws]
    if verbose:
        print(
            f"Player 1 wins: {player1.wins}, Player 2 wins: {player2.wins}, Draws: {draws}"
        )
    player1.reset_wins()
    player2.reset_wins()
    return scores


if __name__ == "__main__":
    import json
    from torch import load
    from .c4_nets import Connect4Net

    p1 = Player()
    with open("connect_four/models/conv_configs.json", "r", encoding="utf-8") as file:
        conv_configs = json.load(file)
    with open("connect_four/models/fc_configs.json", "r", encoding="utf-8") as file:
        fc_configs = json.load(file)

    conv_config = conv_configs[1]
    fc_config = fc_configs[0]

    model1 = Connect4Net(conv_config, fc_config)
    model2 = Connect4Net(conv_config, fc_config)
    model1.load_state_dict(load("connect_four/models/epoch_3.pth"))
    model2.load_state_dict(load("connect_four/models/model1_epoch_7.pth"))

    p2 = NeuralNetPlayer(model1)
    p3 = NeuralNetPlayer(model2)
    play_games(p2, p3, 10000, verbose=True)
