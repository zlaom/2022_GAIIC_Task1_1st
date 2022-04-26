import torch
import torch.nn as nn
import torch.nn.functional as F

class ATTR_ID_MLP(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        image_dim = 2048
        attr_num = 54
        self.image_dropout = nn.Dropout(0.5)
        self.image_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim)
        )
        self.attr_id_linear = nn.Sequential(
            nn.Embedding(attr_num, image_dim // 2),
            nn.Linear(image_dim // 2, image_dim // 2),
            nn.ReLU(),
            nn.Linear(image_dim // 2, image_dim),
            nn.LayerNorm(image_dim)
        )
        
        self.image_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.LayerNorm(image_dim),
            nn.Sigmoid()
        )
        self.attr_id_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.LayerNorm(image_dim),
            nn.Sigmoid()
        )

        self.itm_head = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(),
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