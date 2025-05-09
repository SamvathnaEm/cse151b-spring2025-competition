import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---

class SeparableCausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        
        kt, ky, kx = kernel_size
        dt, dy, dx = dilation
        
        # Causal padding for time
        self.pad_D = (kt - 1) * dt
        self.pad_H = (ky - 1) * dy // 2
        self.pad_W = (kx - 1) * dx // 2
        
        # Depthwise convolution: one filter per input channel
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, groups=in_channels, bias=False
        )
        # Pointwise convolution: 1x1x1 to mix channels
        self.pointwise = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
    
    def forward(self, x):
        if self.pad_W > 0:
            x = F.pad(x, (self.pad_W, self.pad_W, 0, 0, 0, 0), mode='circular')
        if self.pad_H > 0:
            x = F.pad(x, (0, 0, self.pad_H, self.pad_H, 0, 0), mode='constant', value=0)
        if self.pad_D > 0:
            x = F.pad(x, (0, 0, 0, 0, self.pad_D, 0), mode='constant', value=0)
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        
        # Ensure kernel_size and dilation are tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        
        kt, ky, kx = kernel_size  # Temporal, height (lat), width (lon)
        dt, dy, dx = dilation     # Dilation factors
        
        # Causal padding for time dimension: pad left by (kt-1)*dt
        self.pad_D = (kt - 1) * dt
        # Symmetric padding for spatial dimensions
        self.pad_H = (ky - 1) * dy // 2  # Latitude
        self.pad_W = (kx - 1) * dx // 2  # Longitude
        
        # Conv layer with no built-in padding
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=0, dilation=dilation, bias=bias
        )
    
    def forward(self, x):
        # Input shape: (batch, channels, time, lat, lon) = (N, C, D, H, W)
        
        # Pad longitude (W) with circular padding
        if self.pad_W > 0:
            x = F.pad(x, (self.pad_W, self.pad_W, 0, 0, 0, 0), mode='circular')
        # Pad latitude (H) with zero padding
        if self.pad_H > 0:
            x = F.pad(x, (0, 0, self.pad_H, self.pad_H, 0, 0), mode='constant', value=0)
        # Pad time (D) on the left with zeros for causality
        if self.pad_D > 0:
            x = F.pad(x, (0, 0, 0, 0, self.pad_D, 0), mode='constant', value=0)
        
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation_t=1):
        super().__init__()
        dilation = (dilation_t, 1, 1)
        
        self.conv = SeparableCausalConv3d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        
        out += self.skip(identity)
        out = self.relu(out)
        
        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=32,  # Reduced from 64
        depth=3,      # Reduced from 4
        dropout_rate=0.3,  # Increased from 0.2
    ):
        super().__init__()
        
        self.initial = nn.Sequential(
            SeparableCausalConv3d(n_input_channels, init_dim, kernel_size=kernel_size, dilation=1),
            nn.BatchNorm3d(init_dim),
            nn.ReLU(inplace=True),
        )
        
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim
        dilations = [1, 2, 4][:depth]  # Fewer dilations
        
        for i in range(depth):
            dilation_t = dilations[i]
            out_dim = current_dim  # No doubling
            self.res_blocks.append(
                ResidualBlock(current_dim, out_dim, kernel_size, stride=1, dilation_t=dilation_t)
            )
        
        self.dropout = nn.Dropout3d(dropout_rate)
        self.final = nn.Sequential(
            SeparableCausalConv3d(current_dim, current_dim, kernel_size=kernel_size, dilation=1),
            nn.BatchNorm3d(current_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(current_dim, n_output_channels, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.initial(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.dropout(x)
        x = self.final(x)
        
        return x