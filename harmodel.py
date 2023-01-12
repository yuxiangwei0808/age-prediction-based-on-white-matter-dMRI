import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HarmodelEnsemble(nn.Module):
    def __init__(self, in_dim, feature_len, channel=512):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_dim, channel, 7, padding=3, stride=1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(channel, channel*4,  1),
            nn.BatchNorm1d(channel*4, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(channel*4, channel, 1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(in_dim, channel, 1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(channel, channel, 7, padding=3, stride=1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(channel, channel*4, 1),
            nn.BatchNorm1d(channel*4, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(channel*4, channel, 1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(channel, channel, 1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(channel, channel, 7, padding=3, stride=1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(channel, channel*4, 1),
            nn.BatchNorm1d(channel*4, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(channel*4, channel, 1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv1d(channel, channel, 1),
            nn.BatchNorm1d(channel, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_len*channel, 128),
            nn.BatchNorm1d(128, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        y = self.block1(x)
        x = y + self.shortcut1(x)

        y = self.block2(x)
        x = (y + self.shortcut2(x))

        y = self.block3(x)
        x = (y + self.shortcut3(x))

        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out


