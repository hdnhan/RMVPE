import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gru.cpu()

        return self.gru(x.cpu())[0].to(x.device)


class BiLSTM(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lstm(x)[0]
