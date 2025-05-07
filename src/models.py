import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "cnn_lstm":
        model = CNN_LSTM(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_prob: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_prob)

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
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

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
        print(x.shape)
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
        kernel_size=3,
        base_dim=64,
        lstm_hidden=2,
        num_lstm_layers=2
    ):
        super().__init__()
        self.encoder2d = nn.Sequential(
            ResidualBlock(n_input_channels, base_dim),
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True)
        )
        
    # Compute flattened spatial feature size after one 2× downsample
        self.H2 = 48 // 2
        self.W2 = 72  // 2
        self.D  = base_dim * 2
        self.feat_size = self.D * self.H2 * self.W2

        # 2) Temporal model
        self.lstm = nn.LSTM(
            input_size=self.feat_size,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden, self.feat_size)

        # 3) Spatial decoder
        self.decoder2d = nn.Sequential(
            ResidualBlock(self.D, self.D),
            nn.ConvTranspose2d(self.D, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )

        # 4) Final head to\ output variables
        self.head = nn.Conv2d(base_dim, n_output_channels, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Encode each time slice
        seq_feats = []
        for t in range(T):
            f = self.encoder2d(x[:, t])         # → (B, D, H2, W2)
            seq_feats.append(f.view(B, -1))     # → (B, feat_size)

        seq = torch.stack(seq_feats, dim=1)     # → (B, T, feat_size)

        # LSTM + re‐expand
        out_seq, _ = self.lstm(seq)             # → (B, T, lstm_hidden)
        recon = self.fc(out_seq)                # → (B, T, feat_size)

        # Decode per time step
        preds = []
        for t in range(T):
            f2 = recon[:, t].view(B, self.D, self.H2, self.W2)
            p  = self.decoder2d(f2)             # → (B, base_dim, H, W)
            preds.append(self.head(p))          # → (B, n_output, H, W)

        return torch.stack(preds, dim=1).mean(dim=1)
