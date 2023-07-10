from torch import nn

LENBOARD = 3 * 3


class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(LENBOARD, 127),  # input layer
            nn.ReLU(),
            nn.Linear(127, 127),  # hidden layer
            nn.ReLU(),
            nn.Linear(127, 1),  # output layer
        )

    def forward(self, x):
        return self.layers(x)
