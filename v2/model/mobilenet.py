
import torch.nn as nn
from torchvision import models

class model_setup(nn.Module):
    def __init__(self, type, num_classes):
        super(model_setup, self).__init__()
        self.num_classes = num_classes
        if type == 's': # 224
            self.base_model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.base_model.classifier[-1] = nn.Linear(1024, num_classes)
        elif type == 'l': # 224
            self.base_model = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            self.base_model.classifier[-1] = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        if self.num_classes == 1:
            return self.base_model(x).view(-1)
        else:
            return self.base_model(x)
