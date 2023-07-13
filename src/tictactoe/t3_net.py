from torch import nn, from_numpy, argmax  # noqa: E402
from .t3_converter import encode_board


class TicTacToeNet(nn.Module):
    def __init__(self, hidden_size=127):
        super(TicTacToeNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(27, hidden_size),  # 27 because 3x3x3
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 9),
        )

    def forward(self, x):
        return self.layers(x)


def predict_move(model, board):
    encoded_board = encode_board(board.state)
    board_tensor = from_numpy(encoded_board).float()
    board_tensor = board_tensor.view(1, -1)
    output = model(board_tensor)
    predicted_move = argmax(output).item()
    return predicted_move
