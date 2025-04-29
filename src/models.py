import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(
            n_input_channels=len(cfg.data.input_vars),
            n_output_channels=len(cfg.data.output_vars),
            **model_kwargs
        )

    elif cfg.model.type == "cnn_lstm":
        # Split out CNN-related kwargs
        cnn_kwargs = {
            "kernel_size": cfg.model.kernel_size,
            "init_dim": cfg.model.init_dim,
            "depth": cfg.model.depth,
            "dropout_rate": cfg.model.dropout_rate,
        }
        model = CNN_LSTM(
            n_input_channels=len(cfg.data.input_vars),
            n_output_channels=len(cfg.data.output_vars),
            cnn_kwargs=cnn_kwargs,
            cnn_feature_dim=cfg.model.cnn_feature_dim,
            lstm_hidden_dim=cfg.model.lstm_hidden_dim,
            lstm_layers=cfg.model.lstm_layers,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2

        # Final prediction layers
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.dropout(x)
        x = self.final(x)

        return x

class CNN_LSTM(nn.Module):
    def __init__(
        self, 
        n_input_channels, 
        n_output_channels, 
        cnn_kwargs, 
        cnn_feature_dim=128,
        lstm_hidden_dim=256,
        lstm_layers=1
    ):
        super().__init__()

        self.cnn = SimpleCNN(n_input_channels=n_input_channels, n_output_channels=cnn_feature_dim, **cnn_kwargs)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.finalLayer = nn.Linear(lstm_hidden_dim, n_output_channels)

    def forward(self, x):
        BATCH, TIME, CHANNEL, HEIGHT, WIDTH = x.shape
        features = []
        for t in range(TIME):
            cnn_out = self.cnn(x[:, t])
            pooled = self.pool(cnn_out)
            pooled = pooled.squeeze(-1).squeeze(-1)
            features.append(pooled)

        sequence = torch.stack(features, dim=1)

        lstm_out, _ = self.lstm(sequence)

        return self.finalLayer(lstm_out)