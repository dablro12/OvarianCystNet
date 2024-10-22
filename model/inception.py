import torch.nn as nn
from torchvision import models
import torch

class binary_model(nn.Module):
    def __init__(self, type, classes=1):
        super(binary_model, self).__init__()
        if type == 'default':
            self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
        elif type == 'default4ch':  # 299
            self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
            self.base_model.Conv2d_1a_3x3.conv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            with torch.no_grad():
                # 기존 RGB 가중치 복사
                self.base_model.Conv2d_1a_3x3.conv.weight[:, :3, :, :] = self.base_model.Conv2d_1a_3x3.conv.weight[:, :3, :, :]
                # 새 채널은 0으로 초기화
                self.base_model.Conv2d_1a_3x3.conv.weight[:, 3:, :, :].zero_()

        # 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.fc = nn.Linear(2048, classes)

    def forward(self, x):
        outputs = self.base_model(x)
        if isinstance(outputs, torch.Tensor):
            return outputs.squeeze(-1)
        else:
            return outputs.logits.squeeze(-1)


class multi_model(nn.Module):
    def __init__(self, type, classes=3):
        super(multi_model, self).__init__()
        if type == 'default':
            self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
        elif type == 'default4ch':  # 299
            self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
            self.base_model.Conv2d_1a_3x3.conv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            with torch.no_grad():
                # 기존 RGB 가중치 복사
                self.base_model.Conv2d_1a_3x3.conv.weight[:, :3, :, :] = self.base_model.Conv2d_1a_3x3.conv.weight[:, :3, :, :]
                # 새 채널은 0으로 초기화
                self.base_model.Conv2d_1a_3x3.conv.weight[:, 3:, :, :].zero_()

        # 분류기 부분을 다중 클래스 분류에 맞게 변경
        self.base_model.fc = nn.Linear(2048, classes)

    def forward(self, x):
        outputs = self.base_model(x)
        if isinstance(outputs, torch.Tensor):
            return outputs.squeeze(-1)
        else:
            return outputs.logits.squeeze(-1)
