import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from .t3_data_generator import DataGenerator
from .t3_converter import encode_board, encode_moves
from .t3_net import TicTacToeNet
from .t3_self_play import play_games


def data_setup():
    generator = DataGenerator()
    generator.generate_data()
    data = generator.get_training_data()

    encoded_data = np.array([encode_board(board) for board, _ in data])
    encoded_data = encoded_data.reshape(-1, 27)
    encoded_labels = np.array([encode_moves(moves) for _, moves in data])

    """
    print(encoded_data[0])
    print(encoded_data[0].shape)
    print(encoded_labels.shape)
    print(encoded_labels[0])
    input("Press Enter to continue...")
    """

    # Create PyTorch tensors from your data
    data_torch = torch.from_numpy(encoded_data).float()
    labels_torch = torch.from_numpy(encoded_labels).float()

    # Create a Dataset from your input data and labels
    dataset = TensorDataset(data_torch, labels_torch)

    # Split your data into training and validation sets
    total_samples = len(dataset)
    train_samples = int(total_samples * 0.99)
    test_samples = total_samples - train_samples
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_samples, test_samples]
    )

    # Create DataLoaders for your training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def model_setup(hid_size=227, lr=0.01):
    # Choose a loss function and optimizer
    model = TicTacToeNet(hid_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    return model, criterion, optimizer


def train(epochs, model, criterion, optimizer, train_loader, test_loader):
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        # start_time = time.time()
        for x, y in train_loader:
            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)

            loss.backward()

            # Update parameters
            optimizer.step()

            """show = True
            if epoch % 10 == 0 and show and epoch != 0:
                print(outputs[0], y[0])
                print(loss.item())
                predicted_move = torch.argmax(outputs[0]).item()
                print("Predicted move: ", predicted_move)
                ins = input("Press Enter to continue...")
                print(ins)
                if ins == "q":
                    show = False"""
            total_loss += loss.item()
        # end_time = time.time()
        # print(f"Epoch: {epoch+1}, Loss: {total_loss / len(train_loader)}")
        # print(f"Time taken for epoch {epoch+1}: {end_time - start_time}")

    model.eval()

    # Validation loop
    with torch.inference_mode():
        total_val_loss = 0
        for x, y in test_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            total_val_loss += loss.item()

        print(f"Validation Loss: {total_val_loss / len(test_loader)}")
    return model


if __name__ == "__main__":
    train_loader, test_loader = data_setup()
    lrArray = [0.001]
    for epoch in range(13):
        print("Epoch: ", epoch)
        model, criterion, optimizer = model_setup(127, 0.001)
        model = train(1, model, criterion, optimizer, train_loader, test_loader)
        play_games(model, 100)
        play_games(model, 100)
