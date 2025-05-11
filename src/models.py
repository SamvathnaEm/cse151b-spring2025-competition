import torch
import torch.nn as nn
from omegaconf import DictConfig


def periodic_padding(x, pad_width):
    """
    Applies periodic padding along both height (latitude) and width (longitude) dimensions.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch, channels, height, width)
        pad_width (int): Number of pixels to pad on each side for both dimensions
    
    Returns:
        torch.Tensor: Padded tensor
    """
    # Pad width (longitude)
    left_pad = x[:, :, :, -pad_width:]
    right_pad = x[:, :, :, :pad_width]
    x_padded = torch.cat([left_pad, x, right_pad], dim=3)
    
    # Pad height (latitude) with periodic boundary
    top_pad = x_padded[:, :, -pad_width:, :]
    bottom_pad = x_padded[:, :, :pad_width, :]
    x_padded = torch.cat([top_pad, x_padded, bottom_pad], dim=2)
    
    return x_padded


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        """
        A single ConvLSTM cell that combines convolution with LSTM gating.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (int): Number of hidden channels
            kernel_size (int): Size of the convolutional kernel
            bias (bool): Whether to use bias in convolutions
        """
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.pad_width = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # For input, forget, output, and cell gates
            kernel_size=kernel_size,
            padding=0,  # Padding handled manually with periodic_padding
            bias=bias,
        )

    def forward(self, input_tensor, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat([input_tensor, h_prev], dim=1)
        combined_padded = periodic_padding(combined, self.pad_width)
        combined_conv = self.conv(combined_padded)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

    def init_hidden(self, batch_size, height, width):
        """69Initialize hidden and cell states."""
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
        )


class ConvLSTMLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        """A layer of ConvLSTM cells processing a sequence."""
        super(ConvLSTMLayer, self).__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, bias)

    def forward(self, input_seq, hidden_state=None):
        """
        Process the input sequence through the ConvLSTM cell.
        
        Args:
            input_seq (torch.Tensor): Shape (batch, seq_len, channels, height, width)
            hidden_state (tuple, optional): Initial (h, c) states
        
        Returns:
            torch.Tensor: Outputs stacked over sequence length
            tuple: Final (h, c) states
        """
        batch_size, seq_len, _, height, width = input_seq.size()
        if hidden_state is None:
            h, c = self.cell.init_hidden(batch_size, height, width)
        else:
            h, c = hidden_state

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(input_seq[:, t], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)


class ConvLSTM(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, hidden_channels, num_layers, kernel_size):
        """
        ConvLSTM model for spatiotemporal prediction.
        
        Args:
            n_input_channels (int): Number of input channels (e.g., 5 for CO2, SO2, etc.)
            n_output_channels (int): Number of output channels (e.g., 2 for tas, pr)
            hidden_channels (int): Number of hidden channels in ConvLSTM layers
            num_layers (int): Number of ConvLSTM layers
            kernel_size (int): Size of the convolutional kernel
        """
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = n_input_channels if i == 0 else hidden_channels
            self.layers.append(ConvLSTMLayer(in_channels, hidden_channels, kernel_size))
        self.final_conv = nn.Conv2d(hidden_channels, n_output_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the ConvLSTM network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, C_in, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, C_out, H, W)
        """
        batch_size, seq_len, _, height, width = x.size()
        for layer in self.layers:
            x, _ = layer(x)  # x becomes (batch, seq_len, hidden_channels, H, W)
            # Verify dimensions after each layer
            assert x.size(2) == self.layers[0].cell.hidden_channels, f"Channel mismatch: {x.size(2)}"
            assert x.size(3) == height, f"Height mismatch: {x.size(3)}"
            assert x.size(4) == width, f"Width mismatch: {x.size(4)}"
        last_output = x[:, -1, :, :, :]  # Take the last time step
        output = self.final_conv(last_output)  # Map to output channels
        return output


def get_model(cfg: DictConfig):
    """Instantiate the model based on the configuration."""
    model_type = cfg.model.type
    if model_type == "simple_cnn":
        model = SimpleCNN(
            n_input_channels=len(cfg.data.input_vars),
            n_output_channels=len(cfg.data.output_vars),
            kernel_size=cfg.model.kernel_size,
            init_dim=cfg.model.init_dim,
            depth=cfg.model.depth,
            dropout_rate=cfg.model.dropout_rate,
        )
    elif model_type == "conv_lstm":
        model = ConvLSTM(
            n_input_channels=len(cfg.data.input_vars),
            n_output_channels=len(cfg.data.output_vars),
            hidden_channels=cfg.model.hidden_channels,
            num_layers=cfg.model.num_layers,
            kernel_size=cfg.model.kernel_size,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


# Keeping SimpleCNN and ResidualBlock for reference
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
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
    def __init__(self, n_input_channels, n_output_channels, kernel_size=3, init_dim=64, depth=4, dropout_rate=0.2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim
        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:
                current_dim *= 2
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