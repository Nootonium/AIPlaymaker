import pickle
import random
import json
import h5py
import numpy as np
from tqdm import tqdm
from torch import load


from .c4_mcts import C4MCTreeSearch
from .c4_board import C4Board
from .c4_converter import encode_board
from .c4_nets import Connect4Net
from .c4_self_play import Player, MCTSPlayer, NeuralNetPlayer


def simulate_agent_game(agent: Player):
    states = []
    actions = []
    policy_probs = []
    rewards = []
    board = C4Board((6, 7), " " * (6 * 7))

    while board.get_winner() is None:
        state = encode_board(board)
        board, action, q_values = agent.make_move_with_q_values(board)

        states.append(state)
        actions.append(action)
        policy_probs.append(q_values.squeeze().cpu())

    total_moves = len(actions)
    winner = 1 if total_moves % 2 == 1 else 2
    game_outcome = board.get_winner()

    for i in range(total_moves):
        player_for_this_move = 1 if i % 2 == 0 else 2

        if game_outcome == " ":
            rewards.append(0)
        elif player_for_this_move == winner:
            rewards.append(1)
        else:
            rewards.append(-1)

    return (
        np.array(states),
        np.array(actions),
        np.array(policy_probs),
        np.array(rewards),
    )


def simulate_two_agent_game(agent1: Player, agent2: Player):
    states = []
    actions = []
    policy_probs = []
    rewards = []
    board = C4Board((6, 7), " " * (6 * 7))
    current_agent = agent1  # Start with agent1

    while board.get_winner() is None:
        state = encode_board(board)
        board, action, q_values = current_agent.make_move_with_q_values(board)
        states.append(state)
        actions.append(action)
        policy_probs.append(q_values)

        # Switch to the other agent
        current_agent = agent2 if current_agent == agent1 else agent1

    game_outcome = board.get_winner()

    if game_outcome == " ":
        rewards = [0] * len(actions)
    elif game_outcome == "1":
        rewards = [1 if i % 2 == 0 else -1 for i in range(len(actions))]
    else:
        rewards = [-1 if i % 2 == 0 else 1 for i in range(len(actions))]

    if game_outcome == "1":
        agent1.won()
    elif game_outcome == "2":
        agent2.won()

    return (
        np.array(states),
        np.array(actions),
        np.array(policy_probs),
        np.array(rewards),
    )


def generate_agents_data(number_of_games, agent1, agent2):
    states_list, actions_list, policy_probs_list, rewards_list = (
        [],
        [],
        [],
        [],
    )
    for i in tqdm(range(number_of_games)):
        if i % 2 == 0:
            states, actions, policy_probs, rewards = simulate_two_agent_game(
                agent1, agent2
            )
        else:
            states, actions, policy_probs, rewards = simulate_two_agent_game(
                agent2, agent1
            )
        if states.ndim != 4:
            print(f"Skipping game {i} due to incorrect states dimension: {states.ndim}")
            continue
        states_list.append(states)
        actions_list.append(actions)
        policy_probs_list.append(policy_probs)
        rewards_list.append(rewards)

    states = np.concatenate(states_list)
    actions = np.concatenate(actions_list)
    policy_probs = np.concatenate(policy_probs_list)
    rewards = np.concatenate(rewards_list)

    return states, actions, policy_probs, rewards


def generate_agent_data(number_of_games, agent):
    states_list, actions_list, policy_probs_list, rewards_list = (
        [],
        [],
        [],
        [],
    )
    for i in tqdm(range(number_of_games)):
        states, actions, policy_probs, rewards = simulate_agent_game(agent)
        if states.ndim != 4:
            print(f"Skipping game {i} due to incorrect states dimension: {states.ndim}")
            continue
        states_list.append(states)
        actions_list.append(actions)
        policy_probs_list.append(policy_probs)
        rewards_list.append(rewards)

    states = np.concatenate(states_list)
    actions = np.concatenate(actions_list)
    policy_probs = np.concatenate(policy_probs_list)
    rewards = np.concatenate(rewards_list)

    return states, actions, policy_probs, rewards


def simulate_mcts_game(iterations, seen_states):
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


def generate_mcts_data(number_of_games, iterations, seen_states_filename):
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
        states, actions, q_values, all_q_values, all_probs, count = simulate_mcts_game(
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


def save_mcts_data(states, actions, q_values, all_q_values, all_probs, filename):
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


def save_agent_data(filename, states, actions, policy_probs, rewards):
    with h5py.File(filename, "a") as hf:
        # For each data type (states, actions, policy_probs, rewards)
        # If dataset exists, resize and append. If not, create.

        # states
        if "states" not in hf:
            hf.create_dataset(
                "states", data=states, maxshape=(None,) + states.shape[1:]
            )
        else:
            hf["states"].resize((hf["states"].shape[0] + states.shape[0]), axis=0)
            hf["states"][-states.shape[0] :] = states

        # actions
        if "actions" not in hf:
            hf.create_dataset(
                "actions", data=actions, maxshape=(None,) + actions.shape[1:]
            )
        else:
            hf["actions"].resize((hf["actions"].shape[0] + actions.shape[0]), axis=0)
            hf["actions"][-actions.shape[0] :] = actions

        # policy_probs
        if "policy_probs" not in hf:
            hf.create_dataset(
                "policy_probs",
                data=policy_probs,
                maxshape=(None,) + policy_probs.shape[1:],
            )
        else:
            hf["policy_probs"].resize(
                (hf["policy_probs"].shape[0] + policy_probs.shape[0]), axis=0
            )
            hf["policy_probs"][-policy_probs.shape[0] :] = policy_probs

        # rewards
        if "rewards" not in hf:
            hf.create_dataset(
                "rewards", data=rewards, maxshape=(None,) + rewards.shape[1:]
            )
        else:
            hf["rewards"].resize((hf["rewards"].shape[0] + rewards.shape[0]), axis=0)
            hf["rewards"][-rewards.shape[0] :] = rewards


def load_mcts_data(filename):
    with h5py.File(filename, "r") as hf:
        states = hf["states"][:]
        actions = hf["actions"][:]
        q_values = hf["q_values"][:]
        all_q_values_str = hf["all_q_values"][:]
        all_probs_str = hf["all_probs"][:]

    all_q_values = [json.loads(q_value_str) for q_value_str in all_q_values_str]
    all_probs = [json.loads(prob_str) for prob_str in all_probs_str]

    return states, actions, q_values, all_q_values, all_probs


def load_data_from_h5(filename):
    with h5py.File(filename, "r") as hf:
        states = hf["states"][:]
        actions = hf["actions"][:]
        policy_probs = hf["policy_probs"][:]
        rewards = hf["rewards"][:]

    return states, actions, policy_probs, rewards


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


def setup_nn_agent(config_path, model_path):
    with open(config_path, "r", encoding="utf-8") as file:
        configs = json.load(file)
    conv_config = configs["conv_config"]
    fc_config = configs["fc_config"]
    model = Connect4Net(conv_config, fc_config)
    model.load_state_dict(load(model_path))
    nn_player = NeuralNetPlayer(model)
    return nn_player


def setup_mcts_agent(iterations, c_param):
    mcts_player = MCTSPlayer(iterations, c_param)
    return mcts_player


if __name__ == "__main__":
    nn_agent = setup_nn_agent(
        "connect_four/models/model1_config.json",
        "connect_four/models/model1_epoch_7.pth",
    )
    mcts_agent = setup_mcts_agent(500, 0.9)

    states, actions, policy_probs, rewards = generate_agents_data(
        1000, nn_agent, mcts_agent
    )
    save_agent_data(
        "connect_four/data/game_rewards_data.h5", states, actions, policy_probs, rewards
    )
