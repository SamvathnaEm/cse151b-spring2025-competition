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

#CHATGPT ADDITION
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        # we pack all four gates into one conv for efficiency
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        # x:      (B, input_dim,  H,  W)
        # h, c:   (B, hidden_dim, H,  W)
        combined = torch.cat([x, h], dim=1)       # (B, input+hidden, H, W)
        conv_out = self.conv(combined)            # (B, 4*hidden, H, W)
        i, f, o, g = torch.chunk(conv_out, 4, dim=1)
        i = torch.relu(i)
        f = torch.sigmoid(f)
        o = torch.relu(o) #3 Sigmoids before
        g = torch.relu(g) #tanh before
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        cnn_kwargs: dict,
        cnn_feature_dim: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
    ):
        super().__init__()

        self.cnn = SimpleCNN(n_input_channels=n_input_channels, n_output_channels=cnn_feature_dim, **cnn_kwargs)
        

        # 2) One ConvLSTMCell per layer (we’ll stack them in a list for >1 layer)
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(
                input_dim = cnn_feature_dim if layer_idx == 0 else lstm_hidden_dim,
                hidden_dim = lstm_hidden_dim,
                kernel_size = cnn_kwargs.get("kernel_size", 3)
            )
            for layer_idx in range(lstm_layers)
        ])

        # 3) Map from hidden state → final climate variable maps
        self.final_conv = nn.Conv2d(
            in_channels=lstm_hidden_dim,
            out_channels=n_output_channels,
            kernel_size=1
        )

    #Heavily GPT altered
    def forward(self, x):
        """
        x: (B, T, C_in, H, W)
        returns: (B, T, C_out, H, W)
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Will hold h & c for each layer
        h = [torch.zeros(B, cell.hidden_dim, H, W, device=device) for cell in self.cell_list]
        c = [torch.zeros_like(h_l)                                    for h_l in h]

        outputs = []
        for t in range(T):
            # 1) Extract spatial features at time t
            feat = self.cnn(x[:, t])              # (B, cnn_feature_dim, H, W)
            # optionally wrap in checkpoint to save memory:
            # feat = checkpoint(self.cnn, x[:, t])

            # 2) Pass through each ConvLSTM layer
            for layer_idx, cell in enumerate(self.cell_list):
                h[layer_idx], c[layer_idx] = cell(feat, h[layer_idx], c[layer_idx])
                feat = h[layer_idx]               # feed this layer’s h to the next

            # 3) Project to final output map
            out_map = self.final_conv(h[-1])     # (B, n_output_channels, H, W)
            outputs.append(out_map)

        # stack on the time axis → (B, T, C_out, H, W)
        return torch.stack(outputs, dim=1)