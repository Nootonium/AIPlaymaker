from torch import nn, cat
from torch.nn import functional as F


class ParallelConvBlock(nn.Module):
    def __init__(self, configs):
        super(ParallelConvBlock, self).__init__()
        self.blocks = nn.ModuleList(
            [self.build_block(config) for config in configs["layers"]]
        )

    def build_block(self, config):
        return nn.Sequential(
            nn.Conv2d(
                config["in_channels"],
                config["out_channels"],
                kernel_size=config["kernel_size"],
                padding=config["padding"],
            ),
            nn.BatchNorm2d(config["out_channels"]),
            nn.ReLU(),
        )

    def forward(self, x):
        outputs = [block(x) for block in self.blocks]
        return cat(outputs, dim=1)


class Connect4Net(nn.Module):
    def __init__(self, action_space: int, conv_config: dict):
        super(Connect4Net, self).__init__()
        self.conv_block = ParallelConvBlock(conv_config)

        self.fc_block = nn.Sequential(
            nn.Linear(conv_config["fc_input_size"], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

    def forward(self, x):
        x = F.interpolate(x, size=(12, 14), mode="nearest")
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        q_values = self.fc_block(x)
        return q_values
