import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import numpy as np
from .c4_nets import Connect4Net
from .c4_data_generator import load_data


# Set the device to use for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_setup(output_size, learning_rate=0.01, conv_config=None):
    # Choose a loss function and optimizer
    if conv_config is None:
        raise ValueError("conv_config must be provided to model_setup")
    new_model = Connect4Net(output_size, conv_config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(new_model.parameters(), learning_rate)
    return new_model, criterion, optimizer


def setup_training_data(train_batch_size=16, test_batch_size=32):
    states, actions, _, all_qs, _ = load_data("connect_four/data/game_data.h5")

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


def train(epochs, model, criterion, optimizer, train_loader):
    # Train the model
    model.train()
    for epoch in range(epochs):
        for states_batch, q_values_batch in train_loader:
            states_batch = states_batch.to(device)

            q_values_batch = q_values_batch.to(device)

            predicted_q_values = model(states_batch)

            loss = criterion(predicted_q_values, q_values_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "connect_four/models/first.pth")


_out_channels = 8
conv_config = {
    "fc_input_size": 12 * 14 * _out_channels * 3,
    "layers": [
        {
            "in_channels": 2,
            "out_channels": _out_channels,
            "kernel_size": 3,
            "padding": 1,
        },
        {
            "in_channels": 2,
            "out_channels": _out_channels,
            "kernel_size": (1, 3),
            "padding": (0, 1),
        },
        {
            "in_channels": 2,
            "out_channels": _out_channels,
            "kernel_size": (3, 1),
            "padding": (1, 0),
        },
    ],
}


if __name__ == "__main__":
    # x = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])
    # resized_x = F.interpolate(x, size=(9, 9), mode="nearest")

    from .c4_self_play import play_games, MCTSPlayer, NeuralNetPlayer, RandomPlayer

    randomboi = RandomPlayer()
    mcboi = MCTSPlayer(num_iterations=500, c_param=0.9)
    trainData = setup_training_data()
    trs = [0.01, 0.001, 0.0001]
    for stepbro in trs:
        model, criterion, optimizer = model_setup(7, stepbro, conv_config)
        train(13, model, criterion, optimizer, trainData)
        model.eval()
        p2 = NeuralNetPlayer(model)
        scores = play_games(mcboi, p2, 10)
