import torch
import torch.nn as nn

# Import the single source of truth for vocabulary size.
# This guarantees SMSFraudCNN's embedding table is always sized to the actual
# token count — previously the hard-coded default of 1000 wasted ~890 rows and
# would silently diverge from the exported vocab.json.
from edge_layer.data import VOCAB_SIZE as _VOCAB_SIZE


class SMSFraudCNN(nn.Module):
    """
    1D CNN for sequence classification on SMS text.

    Architecture:
      Embedding (VOCAB_SIZE × embed_dim) →
      Conv1d (embed_dim → 16, kernel=3) →
      ReLU →
      AdaptiveMaxPool1d(1) →
      Linear (16 → num_classes)

    vocab_size defaults to the authoritative VOCAB_SIZE exported by
    edge_layer/data.py so training, inference, and the Android on-device
    model all use an identical embedding table.
    """

    def __init__(self, vocab_size: int = _VOCAB_SIZE, embed_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc   = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        x = self.embedding(x)        # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)        # (batch, embed_dim, seq_len)
        x = self.conv(x)             # (batch, 16, seq_len)
        x = self.relu(x)
        x = self.pool(x).squeeze(2)  # (batch, 16)
        return self.fc(x)            # (batch, num_classes)


class CallDetectionMLP(nn.Module):
    """
    Multi-Layer Perceptron for call-pattern fraud detection.

    Features: [duration, hour_of_day, is_contact_saved, times_called_today,
               source_phone_risk_score]
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, num_classes: int = 2):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        return self.fc2(self.relu(self.fc1(x)))
