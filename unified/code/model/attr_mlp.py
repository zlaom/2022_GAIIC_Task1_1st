import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE_ATTR_ID_MLP(nn.Module):
    def __init__(self, attr_num=54, image_dim=2048, dropout=0):
        super().__init__()
        self.image_dropout = nn.Dropout(dropout)
        self.image_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim), nn.LayerNorm(image_dim)
        )
        self.attr_id_linear = nn.Sequential(
            nn.Embedding(attr_num, image_dim // 2),
            nn.Linear(image_dim // 2, image_dim // 2),
            nn.ReLU(),
            nn.Linear(image_dim // 2, image_dim),
            nn.LayerNorm(image_dim),
        )

        self.image_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim), nn.LayerNorm(image_dim), nn.Sigmoid()
        )
        self.attr_id_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim), nn.LayerNorm(image_dim), nn.Sigmoid()
        )

        self.itm_head = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, image, attr_id):

        attr_id = self.attr_id_linear(attr_id)
        image = self.image_dropout(image)
        image = self.image_linear(image)

        features = torch.cat((image, attr_id), dim=-1)
        image_w = self.image_wsa(features)
        attr_id_w = self.attr_id_wsa(features)
        image = F.normalize(image, dim=-1)
        attr_id = F.normalize(attr_id, dim=-1)
        fusion_feature = image * image_w + attr_id * attr_id_w

        logits = self.itm_head(fusion_feature)
        logits = torch.squeeze(logits)

        return logits


class ATTR_ID_MLP2(nn.Module):
    def __init__(self, attr_num=54, image_dim=2048, dropout=0.5):
        super().__init__()
        self.image_dropout = nn.Dropout(dropout)

        self.image_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim), nn.LayerNorm(image_dim)
        )

        self.attr_id_emb = nn.Embedding(attr_num, 64)

        self.match_head = nn.Sequential(
            nn.Linear(image_dim + 64, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, image, attr_id):
        image = self.image_dropout(image)
        attr_id_emb = self.attr_id_emb(attr_id)
        image = self.image_linear(image)
        features = torch.cat((image, attr_id_emb), dim=-1)
        logits = self.match_head(features)
        logits = torch.squeeze(logits)
        return logits


# 改成embedding相加
class ATTR_ID_MLP3(nn.Module):
    def __init__(self, attr_num=54, image_dim=2048, dropout=0.5):
        super().__init__()
        self.image_dropout = nn.Dropout(dropout)

        self.image_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim), nn.LayerNorm(image_dim)
        )

        self.attr_id_emb = nn.Embedding(attr_num, image_dim)

        self.match_head = nn.Sequential(
            nn.Linear(image_dim, 512),
            # nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, image, attr_id):
        image = self.image_dropout(image)
        attr_id_emb = self.attr_id_emb(attr_id)
        image = self.image_linear(image)
        features = image + attr_id_emb
        logits = self.match_head(features)
        logits = torch.squeeze(logits)
        return logits


class AttrIdClassMLP(nn.Module):
    def __init__(self, attr_num, image_dim=2048, dropout=0.5):
        super().__init__()
        self.image_dropout = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, attr_num),
        )

    def forward(self, image):
        image = self.image_dropout(image)
        logits = self.cls_head(image)
        return logits


class CLIP_ATTR_ID_MLP(nn.Module):
    def __init__(self, attr_num=54, image_dim=2048, dropout=0):
        super().__init__()
        self.image_dropout = nn.Dropout(dropout)

        self.image_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU(),
            nn.Linear(image_dim, image_dim // 2),
            nn.LayerNorm(image_dim // 2),
            nn.ReLU(),
            nn.Linear(image_dim // 2, image_dim // 2),
        )

        self.attr_emb = nn.Embedding(attr_num, image_dim)

        self.attr_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU(),
            nn.Linear(image_dim, image_dim // 2),
            nn.LayerNorm(image_dim // 2),
            nn.ReLU(),
            nn.Linear(image_dim // 2, image_dim // 2),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, attr_id):
        image_features = self.image_linear(image)
        attr_emb = self.attr_emb(attr_id)
        attr_features = self.attr_linear(attr_emb)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        attr_features = attr_features / attr_features.norm(dim=-1, keepdim=True)
        return image_features, attr_features, self.logit_scale.exp()
