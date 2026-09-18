"""
model_cnn.py
------------
A 1D-CNN victim for the flow feature vector, used to show that the paper's
findings replicate across architectures (not just the MLP).

DESIGN NOTE ON THE ATTACK SURFACE:
The network accepts the SAME flat input shape as IDSNet: (batch, in_dim). It
reshapes to (batch, 1, in_dim) INTERNALLY. This is deliberate: the adversary
still perturbs the real, flat feature vector, so the constraint spec and the
projection operator in attack.py apply unchanged. Convolution runs over the
feature axis; there is no artificial 2D "image" reshaping of tabular data.

Drop-in: exposes IDSNetCNN with the same (in_dim, n_classes) constructor and
the same forward signature as IDSNet, so preprocess.py / attack.py /
defense_eval.py / run_experiments.py all work without modification.
"""
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
