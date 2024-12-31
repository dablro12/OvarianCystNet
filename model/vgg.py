import torch.nn as nn
from torchvision import models


class binary_model(nn.Module):
    def __init__(self, type, num_classes):
        """
            ref : https://pytorch.org/vision/stable/models/vgg.html 
        """
        super(binary_model, self).__init__()
        if type == 'vgg16': # 224
            self.base_model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
        elif type =='vgg19': # 224
            self.base_model = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1)

        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(4096, 1)
        
    def forward(self, x):
        out = self.base_model(x).view(-1)
        return out  #Sigmoid 처리

class multi_model(nn.Module):
    def __init__(self, type, num_classes):
        """
            ref : https://pytorch.org/vision/stable/models/vgg.html 
        """
        super(multi_model, self).__init__()
        if type == 'vgg16': # 224
            self.base_model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
        elif type =='vgg19': # 224
            self.base_model = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        # out = self.base_model(x).view(-1)
        return self.base_model(x) 
    