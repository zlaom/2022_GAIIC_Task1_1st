from turtle import forward
from matplotlib import image

from numpy import imag
from models.bert import MacBert
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaiicModel(nn.Module):
    def __init__(self, cfg, mode='pretrain'):
        super().__init__()
        self.mode = mode
        self.bert = MacBert(cfg['MAC_BERT'])
        text_dim, image_dim = cfg['TEXT_DIM'], cfg['IMAGE_DIM']
        num_attr = cfg['NUM_ATTR']

        self.bert_linear = nn.Linear(text_dim, image_dim)

        self.wsa_text = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid(),
        )
        self.wsa_image = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid(),
        )

        self.key_logtis = nn.Linear(image_dim, num_attr) # 12个关键节点属性

    def forward(self, text, image):

        text = self.bert(text)
        text = self.bert_linear(text)

        text_norm = torch.nn.functional.normalize(text, p=2, dim=-1)
        image_norm = torch.nn.functional.normalize(image, p=2, dim=-1)

        cos = torch.mul(text_norm, image_norm)
        cos = torch.sum(cos, dim=-1)
        
        if self.mode == 'finetune':
            features = torch.cat((text, image), dim=1)
            text_w = self.wsa_text(features)
            image_w = self.wsa_image(features)
            mul_features = text_norm * text_w + image_norm * image_w
            logits = self.key_logtis(mul_features)
            logits = torch.sigmoid(logits)
            return cos, logits
        
        return cos

class ITM_ALL_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = MacBert(cfg['MAC_BERT'])
        text_dim, image_dim = cfg['TEXT_DIM'], cfg['IMAGE_DIM']
        self.image_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim)
        )
        self.text_linear = nn.Sequential(
            nn.Linear(text_dim, image_dim),
            nn.LayerNorm(image_dim)
        )
        self.concat_linear = nn.Sequential(
            nn.Linear(image_dim*2, image_dim),
            nn.LayerNorm(image_dim)
        )
        self.image_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid()
        )
        self.text_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid()
        )


        self.concat_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid()
        )
        self.fusion_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid()
        )

        self.itm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(image_dim, 2),
        )
            
    def forward(self, image, text):

        text = self.bert(text)
        text = self.text_linear(text)

        image = self.image_linear(image)

        features = torch.cat((image, text), dim=-1)
        image_w = self.image_wsa(features)
        text_w = self.text_wsa(features)
        fusion_feature = image * image_w + text * text_w

        concat_feature = self.concat_linear(features)
        all_feature = torch.cat((fusion_feature, concat_feature), dim=-1)
        fusion_wsa = self.fusion_wsa(all_feature)
        concat_wsa = self.concat_wsa(all_feature)
        all_feature = concat_feature * concat_wsa + fusion_feature * fusion_wsa

        logits = self.itm_head(fusion_feature)

        return logits



class ITM_ATTR_MLP(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        image_dim = 2048
        self.image_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim)
        )
        self.text_linear = nn.Sequential(
            nn.Embedding(80, image_dim),
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim)
        )
        
        self.concat_linear = nn.Sequential(
            nn.Linear(image_dim*2, image_dim),
            nn.LayerNorm(image_dim)
        )

        self.image_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid()
        )
        self.text_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid()
        )
        
        self.concat_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid()
        )
        self.fusion_wsa = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.Sigmoid()
        )

        self.itm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(image_dim, 2),
        )


    def forward(self, image, text):

        
        text = self.text_linear(text)
        image = self.image_linear(image)

        features = torch.cat((image, text), dim=-1)
        image_w = self.image_wsa(features)
        text_w = self.text_wsa(features)
        fusion_feature = image * image_w + text * text_w

        concat_feature = self.concat_linear(features)
        all_feature = torch.cat((fusion_feature, concat_feature), dim=-1)
        fusion_wsa = self.fusion_wsa(all_feature)
        concat_wsa = self.concat_wsa(all_feature)
        all_feature = concat_feature * concat_wsa + fusion_feature * fusion_wsa

        logits = self.itm_head(fusion_feature)

        return logits

class ITM_ATTR_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = MacBert(cfg['MAC_BERT'])
        text_dim, image_dim = cfg['TEXT_DIM'], cfg['IMAGE_DIM']
        self.image_linear = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim)
        )
        self.itm_head = nn.Sequential(
            nn.Linear(text_dim+image_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(inplace=True),
            nn.Linear(text_dim, 2)
        )
            
    def forward(self, image, text):

        text = self.bert(text)
        image = self.image_linear(image)
        features = torch.cat((image, text), dim=-1)
        logits = self.itm_head(features)

        return logits



class BLIP_Model(nn.Module):
    def __init__(self, cfg, mode='pretrain', momentum=0.3):
        super().__init__()


        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))

        self.mode = mode
        self.bert = MacBert(cfg['MAC_BERT'])
        text_dim, image_dim = cfg['TEXT_DIM'], cfg['IMAGE_DIM']
        
        self.image_linear = nn.Linear(image_dim, 256)
        self.bert_linear = nn.Linear(text_dim, 256)

        self.image_text = nn.Linear(image_dim, text_dim)
        self.wsa_text = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.BatchNorm1d(text_dim),
            nn.Tanh(),
        )
        self.wsa_image = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),
            nn.BatchNorm1d(text_dim),
            nn.Tanh(),
        )

        self.itm_head = nn.Linear(text_dim , 2)

    def forward(self, image, text, alpha, mode='train'):
        if mode != 'train':
            text = self.bert(text)
            image_embeds = self.image_text(image)
            vl_embedding = torch.cat([image_embeds, text], dim=-1)
            all_image_embed_norm = F.normalize(image_embeds, dim=-1)
            all_text_embed_norm = F.normalize(text, dim=-1)
            image_w = self.wsa_image(vl_embedding)
            text_w = self.wsa_text(vl_embedding) 
            vl_features = all_image_embed_norm * image_w + all_text_embed_norm * text_w
            
            vl_output = self.itm_head(vl_features)
            return vl_output
        # itc clip learning
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        
        text = self.bert(text)
        image_norm = F.normalize(self.image_linear(image), dim=-1)
        text_norm = F.normalize(self.bert_linear(text), dim=-1)
                         
        

        sim_i2t_m = image_norm @ text_norm.t() / self.temp  
        sim_t2i_m = text_norm @ image_norm.t() / self.temp 

        sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
        sim_targets.fill_diagonal_(1)          

        # sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
        # sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        
                            
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t_m, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i_m, dim=1)*sim_targets,dim=1).mean() 

        loss_itc = (loss_i2t+loss_t2i)/2
        
        # itm
        image_embeds = self.image_text(image)
        bs = image.size(0)
        with torch.no_grad():       
            weights_t2i = F.softmax(sim_t2i_m[:,:bs],dim=1)+1e-4 
            weights_t2i.fill_diagonal_(0)            
            weights_i2t = F.softmax(sim_i2t_m[:,:bs],dim=1)+1e-4  
            weights_i2t.fill_diagonal_(0)  
        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        all_image_embed = torch.cat([image_embeds, image_embeds_neg,image_embeds], dim=0)
        all_text_embed = torch.cat([text, text, text_embeds_neg], dim=0)
        vl_embedding = torch.cat([all_image_embed, all_text_embed], dim=-1)
        all_image_embed_norm = F.normalize(all_image_embed, dim=-1)
        all_text_embed_norm = F.normalize(all_text_embed, dim=-1)
        image_w = self.wsa_image(vl_embedding)
        text_w = self.wsa_text(vl_embedding) 
        vl_features = all_image_embed_norm * image_w + all_text_embed_norm * text_w
        
        vl_output = self.itm_head(vl_features)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_itm, vl_output, itm_labels


    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

