import torch.nn as nn

class ATTR_ID_MLP(nn.Module):
    def __init__(self, attr_num, image_dim=2048, dropout=0.5):
        super().__init__()
        self.image_dropout = nn.Dropout(dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(image_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, attr_num),
        )


    def forward(self, image):
        image = self.image_dropout(image)
        logits = self.cls_head(image)
        return logits