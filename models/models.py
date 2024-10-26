import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F

class CNNModel(torch.nn.Module):
    def __init__(
        self,
        player_dim: int,
        feature_dim: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=1,
                              kernel_size=(1, feature_dim),
                              bias=True)

        self.fc1 = nn.Linear(player_dim * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3)
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x: torch.Tensor):
        home_team = x[:, 0].unsqueeze(1)
        away_team = x[:, 1].unsqueeze(1)

        home_features = F.tanh(self.conv(home_team)).squeeze(3)
        away_features = F.tanh(self.conv(away_team)).squeeze(3)

        home_features = home_features.view(home_features.size(0), -1)
        away_features = away_features.view(away_features.size(0), -1)

        combined_features = torch.cat((home_features, away_features), dim=1)
        x = F.tanh(self.fc1(combined_features))
        x = self.dropout(x)
        x = F.tanh(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits
