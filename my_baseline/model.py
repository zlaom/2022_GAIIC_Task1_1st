import torch
import torch.nn as nn


class MutilTagModel(torch.nn.Module):
    def __init__(self, input_dim=2048, tag_num=80, drop_rate=0.5) -> None:
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.predict_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, tag_num),
            nn.Sigmoid(),
        )

    def forward(self, image_featrue):
        x = self.dropout(image_featrue)
        o = self.predict_head(x)
        return o
