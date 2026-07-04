import torch
import torch.nn as nn

class SMSFraudCNN(nn.Module):
    """
    1D CNN for sequence classification on text.
    Takes tokenized integer sequences of message words.
    """
    def __init__(self, vocab_size=1000, embed_dim=32, num_classes=2):
        super(SMSFraudCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 1D Convolution over sequence length
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x)        # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)        # (batch, embed_dim, seq_len)
        x = self.conv(x)             # (batch, 16, seq_len)
        x = self.relu(x)
        x = self.pool(x).squeeze(2)  # (batch, 16)
        x = self.fc(x)               # (batch, num_classes)
        return x

class CallDetectionMLP(nn.Module):
    """
    Multi-Layer Perceptron evaluating numerical features of a call.
    Features: [duration, hour_of_day, is_contact_saved, times_called_today, source_phone_risk_score]
    """
    def __init__(self, input_dim=5, hidden_dim=16, num_classes=2):
        super(CallDetectionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, input_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
