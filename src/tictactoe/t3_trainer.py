import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from .t3_data_generator import DataGenerator
from .t3_converter import encode_board, encode_moves
from .t3_net import T3Net
from .t3_self_play import play_games


def calculate_accuracy(logits, targets):
    # Convert logits to class probabilities
    probabilities = torch.sigmoid(logits)

    # Find the class with the highest probability
    predicted_classes = torch.argmax(probabilities, dim=1)

    # Find the true class
    true_classes = torch.argmax(targets, dim=1)

    # Check where predicted and true classes match
    correct_predictions = predicted_classes == true_classes

    # Calculate accuracy
    accuracy = correct_predictions.sum().item() / len(correct_predictions)

    return accuracy


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

    # Create DataLoaders for your training and validation datasets
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def model_setup(hid_size=227, lr=0.01):
    # Choose a loss function and optimizer
    model = T3Net(hid_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    return model, criterion, optimizer


def train(epochs, model, criterion, optimizer, train_loader, test_loader):
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        # start_time = time.time()
        for x, y in train_loader:
            optimizer.zero_grad()

            logits_outputs = model(x)
            total_accuracy += calculate_accuracy(logits_outputs, y)
            loss = criterion(logits_outputs, y)

            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item()
        # end_time = time.time()
        print(f"Epoch: {epoch+1}, Loss: {total_loss / len(train_loader)}")
        print(f"Accuracy: {total_accuracy / len(train_loader)}")
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
    n = 216
    lr = 0.0003
    model, criterion, optimizer = model_setup(n, lr)
    for epoch in range(23):
        print(f"Epoch: {epoch+1}")
        model = train(1, model, criterion, optimizer, train_loader, test_loader)
        score = play_games(model, 100)
        """if score[0] == 0:
            torch.save(
                model.state_dict(), f"tictactoe/models/model_{n}_{lr}_{epoch+1}.pth"
            )"""

    # print("Best score: ", best_score)
    # print("Best params: ", best_params)
