import json
import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from .c4_nets import Connect4Net
from .c4_data_generator import load_mcts_data
from .c4_self_play import play_games, MCTSPlayer, NeuralNetPlayer


# Set the device to use for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_setup(new_model, learning_rate=0.01):
    # Choose a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(new_model.parameters(), learning_rate)
    return new_model, criterion, optimizer


def setup_training_data(train_batch_size=16, test_batch_size=32):
    states, actions, _, all_qs, _ = load_mcts_data("connect_four/data/game_data.h5")

    q_values_all = np.zeros((len(all_qs), 7))  # 7 actions in Connect Four
    for i, q_dict in enumerate(all_qs):
        for action, q_value in q_dict.items():
            q_values_all[i, int(action)] = q_value

    # Convert the training data to PyTorch tensors and move them to the defined device
    states_torch = torch.tensor(states, dtype=torch.float).to(device)
    # actions_torch = torch.tensor(actions, dtype=torch.long).to(device)
    q_values_torch = torch.tensor(q_values_all, dtype=torch.float).to(device)

    # Create a Dataset from your input data
    dataset = TensorDataset(states_torch, q_values_torch)

    # Calculate the split index for train/test
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    print("Data loaded.")
    return train_loader


def train(model, criterion, optimizer, train_loader):
    # Train the model
    tot_loss = 0
    model.train()

    for states_batch, q_values_batch in train_loader:
        states_batch = states_batch.to(device)

        q_values_batch = q_values_batch.to(device)

        predicted_q_values = model(states_batch)

        loss = criterion(predicted_q_values, q_values_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        tot_loss += loss.item()
    print(f"Average loss: {tot_loss / len(train_loader)}")
    # Save the trained model
    # torch.save(model.state_dict(), "connect_four/models/first.pth")


def win_ratio(scores, draw_fraction=0.75):
    wins, losses, draws = scores
    total_games = wins + draws + losses
    if total_games == 0:  # To avoid division by zero
        return 0
    return (wins + draw_fraction * draws) / total_games


def evaluate(model, opponent, num_games, verbose=False):
    nn_agent = NeuralNetPlayer(model)
    scores = play_games(nn_agent, opponent, num_games, verbose=verbose)
    return win_ratio(scores)


def train_loop(
    model,
    criterion,
    optimizer,
    train_loader,
    opponent,
    games_to_play=20,
    epochs=13,
    patience=3,
    verbose=False,
    save_models=False,
    save_path="models/",
):
    best_score = 0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        if verbose:
            print(f"Training model for epoch {epoch + 1}/{epochs}")

        train(model, criterion, optimizer, train_loader)
        score = evaluate(model, opponent, games_to_play, verbose=verbose)
        if verbose:
            print(f"Win ratio: {score}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if save_models:
            os.makedirs(save_path, exist_ok=True)
            save_name = f"epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), os.path.join(save_path, save_name))
            if verbose:
                print(f"Model saved as {save_name}")

        if epochs_without_improvement >= patience:
            if verbose:
                print("Stopping early due to lack of improvement")
            break
    return best_epoch, best_score


def test_training(lr: float):
    train_data = setup_training_data()
    mcts_agent = MCTSPlayer(350, 0.9)

    with open("connect_four/models/conv_configs.json", "r", encoding="utf-8") as file:
        conv_configs = json.load(file)
    with open("connect_four/models/fc_configs.json", "r", encoding="utf-8") as file:
        fc_configs = json.load(file)

    results = []

    conv_config = conv_configs[1]
    fc_config = fc_configs[0]

    new_model = Connect4Net(conv_config, fc_config).to(device)
    print(f"Conv: {conv_config['name']}, FC: {fc_config['name']}")
    model, criterion, optimizer = model_setup(new_model, lr)
    epoch, score = train_loop(
        model,
        criterion,
        optimizer,
        train_data,
        mcts_agent,
        verbose=True,
        games_to_play=25,
        save_models=True,
        save_path="connect_four/models/",
    )
    results.append((score, conv_config["name"], fc_config["name"], epoch))

    results.sort(key=lambda x: -x[0])

    print("Ranking:")
    for rank, (score, conv_name, fc_name, epoch) in enumerate(results):
        print(
            f"{rank + 1}. Conv: {conv_name}, FC: {fc_name}, Score: {score}, Epoch: {epoch}"
        )


if __name__ == "__main__":
    test_training(0.0003)
