import pickle
import random
import numpy as np
import h5py
from tqdm import tqdm

from .c4_mcts import C4MCTreeSearch
from .c4_board import C4Board
from .c4_converter import encode_board


def generate_data_mcTreeSearchs(number_of_games, iterations, seen_states):
    # Initialize the board
    states = []
    actions = []
    q_values = []
    board = C4Board((6, 7), " " * (6 * 7))
    for _ in tqdm(range(number_of_games)):
        board = C4Board((6, 7), " " * (6 * 7))
        while board.get_winner() is None:
            mcts = C4MCTreeSearch(board, 0.9)
            mcts.run(iterations)
            best_child = mcts.root.best_child()
            _, action = board.find_move_position(best_child.board.state)

            if best_child.visits == 0:
                q_value = 0
            else:
                q_value = best_child.wins / best_child.visits

            # Record the current game state, action, and Q value
            state = encode_board(board)
            if state.tobytes() in seen_states:
                board = board.with_move(random.choice(board.get_next_possible_moves()))
                continue
            seen_states.add(state.tobytes())

            states.append(state)
            actions.append(action)
            q_values.append(q_value)

            # Perform the action on the game
            board = board.with_move(action)
    states = np.array(states)
    actions = np.array(actions)
    q_values = np.array(q_values)
    return states, actions, q_values


def generate_data_mcTreeSearchsvsRandom(number_of_games, iterations):
    states = []
    actions = []
    q_values = []

    # Play games
    for _ in tqdm(range(number_of_games)):
        # Initialize the board
        board = C4Board((6, 7), " " * (6 * 7))
        while board.get_winner() is None:
            board = board.with_move(random.choice(board.get_next_possible_moves()))
            if board.get_winner() is not None:
                print("Random won")
                print(board)
                break
            mcts = C4MCTreeSearch(board, 0.9)
            mcts.run(iterations)
            best_child = mcts.root.best_child()
            _, action = board.find_move_position(best_child.board.state)

            if best_child.visits == 0:
                q_value = 0
            else:
                q_value = best_child.wins / best_child.visits

            # Record the current game state, action, and Q value
            states.append(encode_board(board))
            actions.append(action)
            q_values.append(q_value)

            # Perform the action on the game
            board = board.with_move(action)

    # Convert lists to numpy arrays for easier use in training
    states = np.array(states)
    actions = np.array(actions)
    q_values = np.array(q_values)

    return states, actions, q_values


def save_data(states, actions, q_values, filename):
    with h5py.File(filename, "a") as hf:
        if "states" not in hf:
            hf.create_dataset(
                "states", data=states, maxshape=(None,) + states.shape[1:]
            )
        else:
            hf["states"].resize((hf["states"].shape[0] + states.shape[0]), axis=0)
            hf["states"][-states.shape[0] :] = states

        if "actions" not in hf:
            hf.create_dataset(
                "actions", data=actions, maxshape=(None,) + actions.shape[1:]
            )
        else:
            hf["actions"].resize((hf["actions"].shape[0] + actions.shape[0]), axis=0)
            hf["actions"][-actions.shape[0] :] = actions

        if "q_values" not in hf:
            hf.create_dataset(
                "q_values", data=q_values, maxshape=(None,) + q_values.shape[1:]
            )
        else:
            hf["q_values"].resize((hf["q_values"].shape[0] + q_values.shape[0]), axis=0)
            hf["q_values"][-q_values.shape[0] :] = q_values


def load_data(filename):
    with h5py.File(filename, "r") as hf:
        states = hf["states"][:]
        actions = hf["actions"][:]
        q_values = hf["q_values"][:]
    return states, actions, q_values


def save_states(seen_states, filename):
    with open(filename, "wb") as f:
        pickle.dump(seen_states, f)


def load_states(filename):
    try:
        with open(filename, "rb") as f:
            seen_states = pickle.load(f)
    except FileNotFoundError:
        seen_states = set()
    return seen_states


if __name__ == "__main__":
    seen_states = load_states("connect_four/data/seen_states.pkl")
    print("Seen states: ", len(seen_states))
    states, actions, q_values = generate_data_mcTreeSearchs(42, 3500, seen_states)
    save_data(states, actions, q_values, "connect_four/data/game_data.h5")
    save_states(seen_states, "connect_four/data/seen_states.pkl")
