import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class MotionNet(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(MotionNet, self).__init__()

        self.MovementAutoencoder_1 = nn.Sequential(
            nn.Conv1d(3, 1, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.MovementAutoencoder_2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.PoseAutoencoder_1 = nn.Sequential(
            nn.Conv1d(3, 1, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.PoseAutoencoder_2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.LuminanceAutoencoder = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.rPPGAutoencoder = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=dropout_rate)

        self.rPPGDecoder = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.DimensionCollapser = nn.Sequential(
            nn.Conv1d(4, 1, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        N, T, C = x.shape
        originalrPPG = x[:, :, 0]
        movement = x[:, :, 1:4]
        pose = x[:, :, 4:7]
        luminance = x[:, :, 7]

        movement = self.MovementAutoencoder_1(movement.transpose(1, 2))
        movement = self.MovementAutoencoder_2(movement.transpose(1, 2))

        pose = self.PoseAutoencoder_1(pose.transpose(1, 2))
        pose = self.PoseAutoencoder_2(pose.transpose(1, 2))

        luminance = self.LuminanceAutoencoder(luminance.view(N, T, 1))

        originalrPPG, _ = self.rPPGAutoencoder(originalrPPG.view((N, 1, T)))
        originalrPPG = originalrPPG.view((N, T, 1))

        embeddings = torch.cat((originalrPPG, movement, pose, luminance), dim=2)
        rPPG, _ = self.rPPGDecoder(embeddings.transpose(1, 2))
        rPPG = self.DimensionCollapser(rPPG).squeeze()

        return rPPG
