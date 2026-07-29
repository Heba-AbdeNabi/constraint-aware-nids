
import torch
import torch.nn as nn


class IDSNetCNN(nn.Module):
    def __init__(self, in_dim, n_classes, channels=(32, 64, 128), k=3, p=0.3):
        super().__init__()
        self.in_dim = in_dim
        convs = []
        c_prev = 1
        for c in channels:
            convs += [
                nn.Conv1d(c_prev, c, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.Dropout(p),
            ]
            c_prev = c
        self.conv = nn.Sequential(*convs)
        # global average pool over the feature axis -> (batch, channels[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, in_dim) -> (batch, 1, in_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)   # (batch, channels[-1])
        return self.head(x)
