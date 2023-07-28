import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from .c4_net import Connect4Net
from .c4_data_generator import load_data

# Set the device to use for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_setup(output_size, learning_rate=0.01):
    # Choose a loss function and optimizer
    new_model = Connect4Net(output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(new_model.parameters(), learning_rate)
    return new_model, criterion, optimizer


def setup_training_data(train_batch_size=16, test_batch_size=32):
    states, actions, q_values = load_data("data.h5")

    # Convert the training data to PyTorch tensors and move them to the defined device
    states_torch = torch.tensor(states, dtype=torch.float).to(device)
    actions_torch = torch.tensor(actions, dtype=torch.long).to(device)
    q_values_torch = torch.tensor(q_values, dtype=torch.float).to(device)

    # Create a Dataset from your input data
    dataset = TensorDataset(states_torch, actions_torch, q_values_torch)

    # Calculate the split index for train/test
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    print("Data loaded.")
    return train_loader


def train(epochs, model, criterion, optimizer, train_loader):
    # Train the model
    model.train()
    for epoch in range(epochs):
        for states_batch, actions_batch, q_values_batch in train_loader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)
            q_values_batch = q_values_batch.to(device)

            predicted_q_values = model(states_batch)

            predicted_q_values_for_actions = predicted_q_values.gather(
                1, actions_batch.unsqueeze(-1)
            ).squeeze(-1)

            loss = criterion(predicted_q_values_for_actions, q_values_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "connect_four/models/first.pth")


if __name__ == "__main__":
    from .c4_self_play import play_games, MCTSPlayer, NeuralNetPlayer

    p1 = MCTSPlayer(3500, 0.9)
    trainData = setup_training_data()
    model, criterion, optimizer = model_setup(7, 0.001)
    for i in range(23):
        train(5, model, criterion, optimizer, trainData)
        model.eval()
        p2 = NeuralNetPlayer(model)
        scores = play_games(p1, p2, 5)
