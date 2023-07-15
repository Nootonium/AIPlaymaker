from torch import nn, from_numpy, argmax, inference_mode, save, load  # noqa: E402
from .t3_converter import encode_board


class T3Net(nn.Module):
    def __init__(self, hidden_size=127):
        super(T3Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(27, hidden_size),  # 27 because 3x3x3
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 27),
            nn.ReLU(),
            nn.Linear(27, 9),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def predict_move(model, board):
    encoded_board = encode_board(board.state)
    board_tensor = from_numpy(encoded_board).float()
    board_tensor = board_tensor.view(1, -1)
    model.eval()
    with inference_mode():
        output = model(board_tensor)
        predicted_move = argmax(output).item()
        return predicted_move


def save_model(model, filename):
    save(model.state_dict(), filename)


def load_model(model, filename):
    model.load_state_dict(load(filename))
    model.eval()
    return model
