from typing import Tuple

import torch
from torch import nn
from .deepunet import DeepUnet0
from .constants import N_MELS, N_CLASS


class BiLSTM(nn.Module):
    def __init__(
        self, input_features: int, hidden_features: int, num_layers: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lstm(x)[0]


class E2E0(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_lstm: int,
        kernel_size: Tuple[int, int],
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        super().__init__()
        self.unet = DeepUnet0(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        assert n_lstm > 0
        self.fc = nn.Sequential(
            BiLSTM(3 * N_MELS, 256, n_lstm),
            nn.Linear(512, N_CLASS),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
