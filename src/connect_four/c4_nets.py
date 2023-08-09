from torch import nn, cat, randn
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
                stride=config["stride"],
            ),
            nn.BatchNorm2d(config["out_channels"]),
            nn.ReLU(),
        )

    def forward(self, x):
        outputs = [block(x) for block in self.blocks]
        """for output in outputs:
            print(output.shape)
            input("Press Enter to continue...")"""
        return cat(outputs, dim=1)


class FCBlock(nn.Module):
    def __init__(self, config, input_size: int):
        super(FCBlock, self).__init__()
        config["layers"][0]["in_features"] = input_size
        layers = []
        for layer_config in config["layers"]:
            layers.append(
                nn.Linear(layer_config["in_features"], layer_config["out_features"])
            )
            activation = layer_config.get("activation")
            if activation is not None:
                layers.append(self._get_activation(activation))
        self.fc_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_block(x)

    def _get_activation(self, activation: str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Activation function {activation} is not supported.")


class Connect4Net(nn.Module):
    def __init__(self, conv_config: dict, fc_config: dict):
        super(Connect4Net, self).__init__()
        self.conv_block = ParallelConvBlock(conv_config)
        fc_input_size = self._find_conv_output_size(conv_config)
        self.fc_block = FCBlock(fc_config, fc_input_size)

    def _find_conv_output_size(self, conv_config):
        dummy_input = randn(1, conv_config["layers"][0]["in_channels"], 12, 14)
        dummy_output = self.conv_block(dummy_input)
        fc_input_size = dummy_output.view(dummy_output.size(0), -1).size(1)
        return fc_input_size

    def forward(self, x):
        x = F.interpolate(x, size=(12, 14), mode="nearest")
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        q_values = self.fc_block(x)
        return q_values
