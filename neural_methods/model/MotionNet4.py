import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class MotionNet(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(MotionNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate)
        )

        self.lstm = nn.LSTM(64, 1, num_layers=2, batch_first=True, dropout=dropout_rate)

    def forward(self, x):
        N, T, C = x.shape
        originalrPPG = x[:, :, 0]

        data = x.transpose(1, 2)
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data).transpose(1, 2)
        noise, _ = self.lstm(data)

        newrPPG = originalrPPG - noise.squeeze()

        return newrPPG
