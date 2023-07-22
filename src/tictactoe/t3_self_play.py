from tqdm import tqdm
from .t3_board import T3Board
from .t3_tree import T3Tree
from .t3_net import predict_move


class Player:
    def make_move(self, board: T3Board) -> T3Board:
        print(board)
        moves = board.get_next_possible_moves()
        move = int(input(f"Enter move {moves}: "))
        return board.make_move(move)


class MinimaxPlayer(Player):
    def make_move(self, board) -> T3Board:
        tree = T3Tree(board)
        _, str_board = tree.get_best_next_move()
        return T3Board(str_board)


class NeuralNetPlayer(Player):
    def __init__(self, model):
        self.model = model

    def make_move(self, board) -> T3Board:
        move = predict_move(self.model, board)
        return board.make_move(move)


def play_game(player1, player2):
    game = T3Board(" " * 9)
    while game.get_winner() is None:
        game = player1.make_move(game)
        if game.get_winner() is not None:
            break
        game = player2.make_move(game)
    return game.get_winner(), game.state


def play_games(player1, player2, num_games=100) -> list[int]:
    scores = [0, 0, 0]
    for _ in tqdm(range(num_games)):
        res, state = play_game(player1, player2)
        scores[["X", "O", " "].index(res)] += 1
    print(f"X wins: {scores[0]}, O wins: {scores[1]}, Draws: {scores[2]}")
    return scores


if __name__ == "__main__":
    from .t3_net import T3Net, load_model

    m1 = T3Net(216)
    m1 = load_model(m1, "tictactoe/models/model_216_0.0003_23.pth")
    nn1 = NeuralNetPlayer(m1)

    player = Player()

    play_games(nn1, player, 10)
    play_games(player, nn1, 10)
