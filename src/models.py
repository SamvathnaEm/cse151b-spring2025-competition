import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    
    # Add output spatial dimensions from data config if model needs them
    # These are assumed to be present in cfg.data, e.g., cfg.data.output_height
    # For this competition, these are fixed (48, 72)
    # You might need to add output_height/width to your data config or handle them here
    model_kwargs["output_height"] = cfg.data.get("output_height", 48) # Default to 48 if not in config
    model_kwargs["output_width"] = cfg.data.get("output_width", 72)   # Default to 72 if not in config

    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "cnnlstm":
        model = CNNLSTM(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    where Swish(x) = x * sigmoid(x) and ⊙ is element-wise multiplication.
    
    For conv layers, we apply this channel-wise.
    """
    def __init__(self, dim):
        super().__init__()
        # Split the dimension to create gate and value projections
        self.gate_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
    
    def forward(self, x):
        gate = self.gate_proj(x)
        value = self.value_proj(x)
        return F.silu(gate) * value  # SiLU is Swish: x * sigmoid(x)


class GeGLU(nn.Module):
    """
    GeGLU activation: GELU(xW + b) ⊙ (xV + c)
    Alternative to SwiGLU that sometimes works better for spatial data.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
    
    def forward(self, x):
        gate = self.gate_proj(x)
        value = self.value_proj(x)
        return F.gelu(gate) * value


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation_type="swiglu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        # Use GroupNorm instead of BatchNorm for better performance with varying batch sizes
        self.gn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        
        # Choose activation function
        if activation_type == "swiglu":
            self.activation1 = SwiGLU(out_channels)
        elif activation_type == "geglu":
            self.activation1 = GeGLU(out_channels)
        else:  # Default to SiLU (Swish)
            self.activation1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.gn2 = nn.GroupNorm(min(32, out_channels), out_channels)
        
        if activation_type == "swiglu":
            self.activation2 = SwiGLU(out_channels)
        elif activation_type == "geglu":
            self.activation2 = GeGLU(out_channels)
        else:
            self.activation2 = nn.SiLU(inplace=True)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), 
                nn.GroupNorm(min(32, out_channels), out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += self.skip(identity)
        out = self.activation2(out)

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
        output_height=None,
        output_width=None,
        activation_type="swiglu",  # New parameter for activation choice
    ):
        super().__init__()
        
        self.activation_type = activation_type

        # Initial convolution to expand channels
        initial_layers = [
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(32, init_dim), init_dim),
        ]
        
        # Add appropriate activation
        if activation_type == "swiglu":
            initial_layers.append(SwiGLU(init_dim))
        elif activation_type == "geglu":
            initial_layers.append(GeGLU(init_dim))
        else:
            initial_layers.append(nn.SiLU(inplace=True))
            
        self.initial = nn.Sequential(*initial_layers)

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim, activation_type=activation_type))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2
        
        self.feature_dim_after_res_blocks = current_dim

        # Conditionally create final layers and dropout
        if n_output_channels > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
            
            # Build final convolution layers
            final_layers = [
                nn.Conv2d(self.feature_dim_after_res_blocks, self.feature_dim_after_res_blocks // 2, 
                         kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(min(32, self.feature_dim_after_res_blocks // 2), self.feature_dim_after_res_blocks // 2),
            ]
            
            # Add activation
            if activation_type == "swiglu":
                final_layers.append(SwiGLU(self.feature_dim_after_res_blocks // 2))
            elif activation_type == "geglu":
                final_layers.append(GeGLU(self.feature_dim_after_res_blocks // 2))
            else:
                final_layers.append(nn.SiLU(inplace=True))
                
            # Final output layer (no activation)
            final_layers.append(nn.Conv2d(self.feature_dim_after_res_blocks // 2, n_output_channels, kernel_size=1))
            
            self.final_convs = nn.Sequential(*final_layers)
        else:
            # This implies feature extraction mode (n_output_channels <= 0)
            self.dropout = None
            self.final_convs = None
            
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with appropriate schemes for different layer types."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        x = self.initial(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return x # Shape: (B, self.feature_dim_after_res_blocks, H, W)

    def forward(self, x):
        features = self.extract_features(x)
        
        # Only apply dropout and final_convs if they exist (i.e., not in feature extraction mode)
        if self.final_convs is not None and self.dropout is not None:
            out = self.dropout(features)
            out = self.final_convs(out)
            return out
        else:
            # In feature extraction mode (e.g., when called by CNNLSTM via extract_features)
            # or if SimpleCNN's forward is called directly but it was configured as extractor.
            return features


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        output_height,
        output_width,
        cnn_kernel_size=3,
        cnn_init_dim=64,
        cnn_depth=4,
        cnn_dropout_rate=0.2,
        cnn_activation_type="swiglu",
        lstm_hidden_dim=256,
        n_lstm_layers=2,
        lstm_dropout=0.2,
        use_attention=False,  # Option to add attention mechanism
        attention_heads=8,  # Number of attention heads
        bidirectional=False,  # Whether to use bidirectional LSTM
        adaptive_pool_size=4,  # Size for pooling (now deterministic)
        feature_projection_factor=1.0,  # Scaling factor for feature projection
    ):
        super().__init__()
        self.n_output_channels = n_output_channels
        self.output_height = output_height
        self.output_width = output_width
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.adaptive_pool_size = adaptive_pool_size

        self.cnn = SimpleCNN(
            n_input_channels=n_input_channels,
            n_output_channels=-1,  # Feature extraction mode
            kernel_size=cnn_kernel_size,
            init_dim=cnn_init_dim,
            depth=cnn_depth,
            dropout_rate=cnn_dropout_rate,
            activation_type=cnn_activation_type,
        )
        
        cnn_feature_channels = self.cnn.feature_dim_after_res_blocks

        # Use deterministic pooling instead of adaptive pooling
        # Calculate the appropriate kernel sizes and strides for deterministic pooling
        # Assuming the CNN output maintains the same spatial dimensions as input (48x72)
        input_h, input_w = 48, 72  # Known from data config
        target_h, target_w = adaptive_pool_size, adaptive_pool_size
        
        # Calculate kernel sizes and strides for deterministic pooling
        # We'll use average pooling with calculated kernel sizes
        kernel_h = input_h // target_h
        kernel_w = input_w // target_w
        stride_h = kernel_h
        stride_w = kernel_w
        
        # Handle any remainder by adjusting the kernel size slightly
        if input_h % target_h != 0:
            kernel_h = input_h // target_h + 1
            stride_h = input_h // target_h
        if input_w % target_w != 0:
            kernel_w = input_w // target_w + 1
            stride_w = input_w // target_w
            
        self.deterministic_pool = nn.AvgPool2d(
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=0
        )
        
        # Calculate the actual output size after pooling (might be slightly different from target)
        pooled_h = (input_h - kernel_h) // stride_h + 1
        pooled_w = (input_w - kernel_w) // stride_w + 1
        pooled_feature_size = cnn_feature_channels * pooled_h * pooled_w
        
        # Store actual pooled dimensions for debugging
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        
        # Compute projected feature size with scaling factor
        projected_feature_size = int(lstm_hidden_dim * feature_projection_factor)
        
        # Optional: Add a projection layer to reduce dimensionality
        self.feature_projection = nn.Sequential(
            nn.Linear(pooled_feature_size, projected_feature_size),
            nn.SiLU(),
            nn.Dropout(cnn_dropout_rate),
            nn.Linear(projected_feature_size, lstm_hidden_dim),  # Project to LSTM input size
            nn.SiLU(),
            nn.Dropout(cnn_dropout_rate)
        )
        
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,  # Now using projected features
            hidden_size=lstm_hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if n_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Adjust LSTM output size if bidirectional
        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        
        # Optional attention mechanism for better temporal modeling
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=attention_heads,
                dropout=lstm_dropout,
                batch_first=True
            )
        
        # Enhanced decoder with multiple layers and residual connections
        self.fc_decoder = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim * 2),
            nn.SiLU(),
            nn.Dropout(lstm_dropout),
            nn.Linear(lstm_output_dim * 2, lstm_output_dim * 2),
            nn.SiLU(), 
            nn.Dropout(lstm_dropout),
            nn.Linear(lstm_output_dim * 2, n_output_channels * output_height * output_width)
        )
        
        # Add a residual projection for better gradient flow
        self.residual_projection = nn.Linear(lstm_output_dim, n_output_channels * output_height * output_width)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights properly for LSTM and linear layers."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize linear layers with proper initialization for SiLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, h_0=None, c_0=None):
        batch_size, seq_len, C_in, H_in, W_in = x.shape
        
        # Process each timestep through CNN
        cnn_input = x.reshape(batch_size * seq_len, C_in, H_in, W_in)
        cnn_features = self.cnn.extract_features(cnn_input)
        
        # Pool and project features using deterministic pooling
        pooled_features = self.deterministic_pool(cnn_features)
        pooled_features = pooled_features.reshape(batch_size * seq_len, -1)
        projected_features = self.feature_projection(pooled_features)
        
        # Reshape for LSTM: (batch_size, seq_len, feature_dim)
        lstm_input = projected_features.reshape(batch_size, seq_len, -1)
        
        # LSTM processing
        if h_0 is not None and c_0 is not None:
            lstm_out, (h_n, c_n) = self.lstm(lstm_input, (h_0, c_0))
        else:
            lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Optional attention mechanism
        if self.use_attention:
            lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Decode to output format with residual connection
        decoder_input = lstm_out.reshape(batch_size * seq_len, -1)
        
        # Main decoder path
        decoded_output = self.fc_decoder(decoder_input)
        
        # Residual connection - add direct projection of LSTM output
        residual_output = self.residual_projection(decoder_input)
        final_output = decoded_output + residual_output
        
        output = final_output.view(
            batch_size, seq_len, self.n_output_channels, self.output_height, self.output_width
        )
        
        return output