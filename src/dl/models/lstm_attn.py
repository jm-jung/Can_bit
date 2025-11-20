"""
LSTM + Attention model for time series classification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttentionModel(nn.Module):
    """
    LSTM with Attention mechanism for binary classification.
    
    Architecture:
    - 2-layer LSTM
    - Attention layer over LSTM outputs
    - MLP for final classification
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize LSTM + Attention model.

        Args:
            input_size: Number of input features (feature_dim)
            hidden_size: LSTM hidden size (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.2)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # MLP for final classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim)

        Returns:
            Probability tensor of shape (batch, 1) with values in (0, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention mechanism
        # Compute attention scores for each timestep
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)

        # Final classification
        logit = self.classifier(context)  # (batch, 1)

        # NOTE: 여기서는 sigmoid 안 씌우고 raw logit만 반환한다.
        # 손실 계산은 BCEWithLogitsLoss가 내부에서 sigmoid까지 처리한다.
        return logit

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
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> LSTMAttentionModel:
        """
        Load model from state_dict.

        Args:
            path: Path to saved model
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            device: Device to load model on (default: auto-detect)

        Returns:
            Loaded model instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model

