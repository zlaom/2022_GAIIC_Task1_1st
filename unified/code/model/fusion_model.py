import torch
import torch.nn as nn


class FusionTitleMlp(nn.Module):
    def __init__(self, key_attr_num=13):
        super().__init__()
        self.fusion_linear1 = nn.Sequential(
            nn.Linear(key_attr_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self.fusion_linear2 = nn.Sequential(
            nn.Linear(key_attr_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, origin_predict, mask):
        title_origin_predict = origin_predict[:, [1]]
        attr_origin_predict = origin_predict[:, 1:]
        attr_mask = mask[:, 1:]
        attr_num = torch.sum(attr_mask, dim=-1, keepdim=True)
        # 属性预测标准化
        attr_origin_predict = attr_origin_predict / attr_num

        # 1次修正
        input_data = torch.cat((title_origin_predict, attr_origin_predict), dim=-1)
        new_title_predict = self.fusion_linear1(input_data) + title_origin_predict

        # 2次修正
        input_data = torch.cat((new_title_predict, attr_origin_predict), dim=-1)
        new_title_predict = self.fusion_linear2(input_data) + new_title_predict

        # 裁剪
        new_title_predict = torch.clamp(new_title_predict, 0, 1)

        return new_title_predict


class FusionAttrMlp(nn.Module):
    def __init__(self, key_attr_num=13):
        super().__init__()

        self.fusion_linear1 = nn.Sequential(
            nn.Linear(key_attr_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, key_attr_num - 1),
            nn.Tanh(),
        )

        self.fusion_linear2 = nn.Sequential(
            nn.Linear(key_attr_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, key_attr_num - 1),
            nn.Tanh(),
        )

    def forward(self, origin_predict, mask):
        title_origin_predict = origin_predict[:, [1]]
        attr_origin_predict = origin_predict[:, 1:]
        attr_mask = mask[:, 1:]
        attr_num = torch.sum(attr_mask, dim=-1, keepdim=True)

        # 属性预测标准化
        attr_origin_predict = attr_origin_predict / attr_num

        # 1次修正
        input_data = torch.cat((title_origin_predict, attr_origin_predict), dim=-1)
        new_attr_predict = attr_origin_predict + self.fusion_linear1(input_data)
        new_attr_predict[attr_mask == 0] = 0

        # 2次修正
        input_data = torch.cat((title_origin_predict, attr_origin_predict), dim=-1)
        new_attr_predict = new_attr_predict + self.fusion_linear1(input_data)
        new_attr_predict[attr_mask == 0] = 0

        # 裁剪
        new_attr_predict = torch.clamp(new_attr_predict, 0, 1)

        return new_attr_predict


if __name__ == "__main__":
    origin_predict = torch.randn((128, 13))
    mask = torch.randn((128, 13)) > 0.5
    model = FusionTitleMlp()
    predict = model(origin_predict, mask)
    print(predict)

    origin_predict = torch.randn((128, 13))
    mask = torch.randn((128, 13)) > 0.5
    model = FusionAttrMlp()
    predict = model(origin_predict, mask)
    print(predict)
