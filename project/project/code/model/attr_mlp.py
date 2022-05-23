import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FinalCatModel(nn.Module):
    def __init__(self, attr_num=80, image_dim=2048, dropout=0):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_dim, image_dim),
            nn.BatchNorm1d(image_dim),
        )

        self.attr_id_encoder = nn.Sequential(
            nn.Embedding(attr_num, image_dim),
            nn.BatchNorm1d(image_dim),
        )

        self.a_id2image = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid(),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(2 * image_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, image, attr_id):
        attr = self.attr_id_encoder(attr_id)
        image = self.image_encoder(image)

        a_id2image = self.a_id2image(attr)

        features = torch.cat((image * a_id2image, attr), dim=-1)

        logits = self.fusion_head(features)
        logits = torch.squeeze(logits, dim=1)

        return logits

