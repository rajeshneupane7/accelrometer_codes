import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers, n_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers, n_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_forward = h[-2]
        h_backward = h[-1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(h_cat)

class CNN1DClassifier(nn.Module):
    def __init__(self, n_features, n_filters, kernel_size, n_classes):
        super().__init__()
        self.conv = nn.Conv1d(n_features, n_filters, kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TransformerClassifier(nn.Module):
    def __init__(self, n_features, n_heads, n_layers, n_classes, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, dim_feedforward)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=n_heads,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(dim_feedforward, n_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)