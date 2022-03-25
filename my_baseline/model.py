import torch
import torch.nn as nn


class MutilTagModel(torch.nn.Module):
    def __init__(self, input_dim=2048, tag_num=80) -> None:
        super().__init__()
        self.predict_head = nn.Sequential(nn.Linear(input_dim, input_dim//2), nn.ReLU(), nn.Linear(input_dim//2, tag_num), nn.Sigmoid())
    def forward(self, image_featrue):
        return self.predict_head(image_featrue)