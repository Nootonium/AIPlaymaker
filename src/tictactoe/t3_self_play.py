from tqdm import tqdm
from .t3_board import T3Board
from .t3_tree import T3Tree
from .t3_net import predict_move


def play_game(inmodel):
    # Initialize a new game
    game = T3Board(" " * 9)

    # Game loop
    while game.get_winner() is None:
        # Let minimax make a move
        # calculate time taken
        tree = T3Tree(game)
        movedata = tree.get_best_next_move()
        new_board = movedata.get("post_move_board")
        game = T3Board(new_board)

        # Check if game is over after minimax's move
        if game.get_winner() is not None:
            break

        move = predict_move(inmodel, game)
        game.make_move(move)
    return game.get_winner(), game.state


def play_games(inmodel, num_games=100):
    scores = [0, 0, 0]
    for _ in tqdm(range(num_games)):
        res, state = play_game(inmodel)
        scores[["X", "O", " "].index(res)] += 1
        if res == "X":
            # print("state: ", state)
            pass

    print(f"X wins: {scores[0]}, O wins: {scores[1]}, Draws: {scores[2]}")
