"""
Temporal Convolutional Network (TCN) model for time series classification.

This model uses 1D dilated causal convolutions for time series classification.
Supports 3-class classification (FLAT/LONG/SHORT):
- Class 0: FLAT (ambiguous/neutral zone)
- Class 1: LONG (strong upward movement expected)
- Class 2: SHORT (strong downward movement expected)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl.data.labels import LstmClassIndex


class TemporalBlock(nn.Module):
    """
    Temporal block with dilated causal convolution.
    
    Each block consists of:
    - Dilated causal convolution
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection (if input/output channels match)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Padding to ensure causal convolution (only past values)
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        
        # Weight normalization
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.conv2 = nn.utils.weight_norm(self.conv2)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection (if channels match)
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.residual is not None:
            nn.init.kaiming_normal_(self.residual.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        residual = x
        
        # First convolution
        out = self.conv1(x)
        # Remove padding from the right (causal convolution)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for 3-class classification.
    
    Architecture:
    - Multiple temporal blocks with increasing dilation rates
    - Global average pooling
    - MLP for final classification (3-class softmax)
    
    Output:
    - Raw logits of shape (batch, 3) for CrossEntropyLoss
    - Apply softmax during inference to get probabilities
    """
    
    def __init__(
        self,
        input_size: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        """
        Initialize TCN model.
        
        Args:
            input_size: Number of input features (feature dimension)
            num_channels: List of channel sizes for each temporal block layer
                          Default: [64, 64, 64, 64] (4 layers)
            kernel_size: Convolution kernel size (default: 3)
            dropout: Dropout rate (default: 0.2)
            num_classes: Number of output classes (default: 3 for FLAT/LONG/SHORT)
        """
        super().__init__()
        
        if num_channels is None:
            num_channels = [64, 64, 64, 64]
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Build temporal blocks
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        
        self.temporal_blocks = nn.ModuleList(layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
               or (batch, input_size, seq_len) if already transposed
               
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Input shape: (batch, seq_len, input_size)
        # TCN expects: (batch, channels, seq_len)
        if x.dim() == 3 and x.size(1) != self.input_size:
            # Assume (batch, seq_len, input_size) -> transpose to (batch, input_size, seq_len)
            x = x.transpose(1, 2)
        
        # Temporal blocks
        for block in self.temporal_blocks:
            x = block(x)
        
        # Global average pooling: (batch, channels, seq_len) -> (batch, channels, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch, channels, 1) -> (batch, channels)
        x = x.squeeze(-1)
        
        # Classifier: (batch, channels) -> (batch, num_classes)
        logits = self.classifier(x)
        
        # NOTE: 여기서는 softmax 안 씌우고 raw logits만 반환한다.
        # 손실 계산은 CrossEntropyLoss가 내부에서 softmax까지 처리한다.
        # 추론 시점에는 F.softmax(logits, dim=-1)를 적용하여 확률을 얻는다.
        return logits
    
    def save_model(self, path: str | Path) -> None:
        """
        Save model state_dict to file.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_model(
        cls,
        path: str | Path,
        input_size: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
        device: Optional[torch.device] = None,
    ) -> TCNModel:
        """
        Load model from state_dict.
        
        Args:
            path: Path to saved model
            input_size: Number of input features
            num_channels: List of channel sizes (must match training)
            kernel_size: Convolution kernel size (must match training)
            dropout: Dropout rate (must match training)
            num_classes: Number of output classes (must match training)
            device: Device to load model on (default: auto-detect)
            
        Returns:
            Loaded model instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if num_channels is None:
            num_channels = [64, 64, 64, 64]
        
        model = cls(
            input_size=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            num_classes=num_classes,
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model

