import pickle
import random
import json
import h5py
import numpy as np
from tqdm import tqdm


from .c4_mcts import C4MCTreeSearch
from .c4_board import C4Board
from .c4_converter import encode_board


def simulate_game(iterations, seen_states):
    # Initialize the board
    states = []
    actions = []
    q_values = []
    all_q_values = []  # a list of dictionaries
    all_probs = []  # a list of dictionaries
    count = 0
    board = C4Board((6, 7), " " * (6 * 7))
    while board.get_winner() is None:
        state = encode_board(board)
        if state.tobytes() in seen_states:
            board = board.with_move(random.choice(board.get_next_possible_moves()))
            count += 1
            continue

        mcts = C4MCTreeSearch(board, 0.9)
        new_board = mcts.run(iterations)
        _, action = board.find_move_position(new_board.state)

        seen_states.add(state.tobytes())
        states.append(state)
        actions.append(action)
        qvs = mcts.root.get_q_values()
        q_values.append(qvs[action])
        all_q_values.append(qvs)
        all_probs.append(mcts.root.get_probs())

        # Perform the action on the game
        board = board.with_move(action)

    states = np.array(states)
    actions = np.array(actions)
    q_values = np.array(q_values)

    return states, actions, q_values, all_q_values, all_probs, count


def generate_data(number_of_games, iterations, seen_states_filename):
    seen_states = load_states(seen_states_filename)
    print("Seen states: ", len(seen_states))

    states_list, actions_list, q_values_list, all_q_values_list, all_probs_list = (
        [],
        [],
        [],
        [],
        [],
    )
    all_count = 0
    for i in tqdm(range(number_of_games)):
        states, actions, q_values, all_q_values, all_probs, count = simulate_game(
            iterations, seen_states
        )
        if states.ndim != 4:
            print(f"Skipping game {i} due to incorrect states dimension: {states.ndim}")
            continue
        states_list.append(states)
        actions_list.append(actions)
        q_values_list.append(q_values)
        all_q_values_list.append(all_q_values)
        all_probs_list.append(all_probs)
        all_count += count

    states = np.concatenate(states_list)
    actions = np.concatenate(actions_list)
    q_values = np.concatenate(q_values_list)
    all_q_values = [q for sublist in all_q_values_list for q in sublist]
    all_probs = [p for sublist in all_probs_list for p in sublist]
    print("Number of moves skipped: ", all_count)
    print("Number of moves: ", len(states))
    save_states(seen_states, seen_states_filename)

    return states, actions, q_values, all_q_values, all_probs


def save_data(states, actions, q_values, all_q_values, all_probs, filename):
    all_q_values_str = [json.dumps(q_value_dict) for q_value_dict in all_q_values]
    all_probs_str = [json.dumps(prob_dict) for prob_dict in all_probs]
    dt = h5py.string_dtype(encoding="utf-8")
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

        if "all_q_values" not in hf:
            maxshape = (None,)
            data = np.array(all_q_values_str, dtype=dt)
            hf.create_dataset("all_q_values", data=data, maxshape=maxshape)
        else:
            old_len = hf["all_q_values"].shape[0]
            new_len = old_len + len(all_q_values_str)
            hf["all_q_values"].resize((new_len,))
            hf["all_q_values"][old_len:] = np.array(all_q_values_str, dtype=dt)

        if "all_probs" not in hf:
            maxshape = (None,)
            data = np.array(all_probs_str, dtype=dt)
            hf.create_dataset("all_probs", data=data, maxshape=maxshape)
        else:
            old_len = hf["all_probs"].shape[0]
            new_len = old_len + len(all_probs_str)
            hf["all_probs"].resize((new_len,))
            hf["all_probs"][old_len:] = np.array(all_probs_str, dtype=dt)


def load_data(filename):
    with h5py.File(filename, "r") as hf:
        states = hf["states"][:]
        actions = hf["actions"][:]
        q_values = hf["q_values"][:]

        all_q_values_str = hf["all_q_values"][:]
        all_probs_str = hf["all_probs"][:]

    all_q_values = [json.loads(q_value_str) for q_value_str in all_q_values_str]
    all_probs = [json.loads(prob_str) for prob_str in all_probs_str]

    return states, actions, q_values, all_q_values, all_probs


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
    """ss, sq, sa, sall_q, sall_p = load_data("connect_four/data/game_data.h5")
    print(ss.shape, sa.shape, sq.shape, len(sall_q), len(sall_p))
    input("This will overwrite the data files. Press enter to continue...")"""
    for _ in range(10):
        s, a, q, all_q, all_p = generate_data(
            100, 3500, "connect_four/data/seen_states.pkl"
        )
        save_data(s, a, q, all_q, all_p, "connect_four/data/game_data.h5")
