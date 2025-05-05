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

        self.se = SqueezeExcite(out_channels)  # 👈 Added SE module

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)  # 👈 Apply SE before adding skip

        out += self.skip(identity)
        out = self.relu(out)

        return out
    
class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


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

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolutional layers in self.initial
        for module in self.initial:
            if isinstance(module, nn.Conv2d):
                # Use He initialization for ReLU (recommended)
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # Alternative: Use Xavier for consistency
                # nn.init.xavier_uniform_(module.weight)
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)

        # Initialize convolutional layers in self.res_blocks
        for res_block in self.res_blocks:
            for module in res_block.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    # Alternative: nn.init.xavier_uniform_(module.weight)

        # Initialize convolutional layers in self.final
        for module in self.final:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # Alternative: nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.dropout(x)
        x = self.final(x)

        return x

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
        self.norm_i = nn.LayerNorm([hidden_dim, 1, 1])
        self.norm_f = nn.LayerNorm([hidden_dim, 1, 1])
        self.norm_o = nn.LayerNorm([hidden_dim, 1, 1])
        self.norm_g = nn.LayerNorm([hidden_dim, 1, 1])

        # Initialize weights
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            # Optional: Set a positive bias for forget gate (indices hidden_dim:2*hidden_dim)
            # self.conv.bias.data[self.hidden_dim:2*self.hidden_dim].fill_(0.1)

    def forward(self, x, h, c):
        # x:      (B, input_dim,  H,  W)
        # h, c:   (B, hidden_dim, H,  W)
        combined = torch.cat([x, h], dim=1)       # (B, input+hidden, H, W)
        conv_out = self.conv(combined)            # (B, 4*hidden, H, W)
        i, f, o, g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
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
        
        # One ConvLSTMCell per layer
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(
                input_dim=cnn_feature_dim if layer_idx == 0 else lstm_hidden_dim,
                hidden_dim=lstm_hidden_dim,
                kernel_size=cnn_kwargs.get("kernel_size", 3)
            )
            for layer_idx in range(lstm_layers)
        ])

        # Final 1x1 conv to map hidden state to output
        self.final_conv = nn.Conv2d(
            in_channels=lstm_hidden_dim,
            out_channels=n_output_channels,
            kernel_size=1
        )

        # Initialize weights for final_conv
        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        """
        x: (B, T, C_in, H, W)
        returns: (B, T, C_out, H, W)
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Initialize h and c for each layer
        h = [torch.zeros(B, cell.hidden_dim, H, W, device=device) for cell in self.cell_list]
        c = [torch.zeros_like(h_l) for h_l in h]

        outputs = []
        for t in range(T):
            # Extract spatial features at time t
            feat = self.cnn(x[:, t])  # (B, cnn_feature_dim, H, W)

            # Pass through each ConvLSTM layer
            for layer_idx, cell in enumerate(self.cell_list):
                h[layer_idx], c[layer_idx] = cell(feat, h[layer_idx], c[layer_idx])
                feat = h[layer_idx]  # Feed this layer’s h to the next

            # Project to final output map
            out_map = self.final_conv(h[-1])  # (B, n_output_channels, H, W)
            outputs.append(out_map)

        # Stack on the time axis → (B, T, C_out, H, W)
        return torch.stack(outputs, dim=1)