
import torch.nn as nn
from torchvision import models

class binary_model(nn.Module):
    def __init__(self, type):
        super(binary_model, self).__init__()
        if type == 's': # 224
            self.base_model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.base_model.classifier[-1] = nn.Linear(1024, 1)
        elif type == 'l': # 224
            self.base_model = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            self.base_model.classifier[-1] = nn.Linear(1280, 1)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        
    def forward(self, x):
        return self.base_model(x).view(-1)

class multi_model(nn.Module):
    def __init__(self, type):
        super(multi_model, self).__init__()
        if type == 's': # 224
            self.base_model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.base_model.classifier[-1] = nn.Linear(1024, 3)
        elif type == 'l': # 224
            self.base_model = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            self.base_model.classifier[-1] = nn.Linear(1280, 3)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        
    def forward(self, x):
        return self.base_model(x)