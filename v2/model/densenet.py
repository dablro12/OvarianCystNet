import torch
import torch.nn as nn 
from torchvision import models

class model_setup(nn.Module):
    """ ref : https://pytorch.org/vision/stable/models/densenet.html """
    def __init__(self, type, num_classes):
        super(model_setup, self).__init__()
        self.num_classes = num_classes
        if type == 'densenet121': # 224
            self.base_model = models.densenet121(weights = models.DenseNet121_Weights.IMAGENET1K_V1)
        elif type == 'densenet161': # 224
            self.base_model = models.densenet161(weights = models.DenseNet161_Weights.IMAGENET1K_V1)
        elif type == 'densenet169': # 224
            self.base_model = models.densenet169(weights = models.DenseNet169_Weights.IMAGENET1K_V1)
        elif type == 'densenet201': # 224
            self.base_model = models.densenet201(weights = models.DenseNet201_Weights.IMAGENET1K_V1)
            
        self.base_model.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        if self.num_classes == 1:
            return self.base_model(x).view(-1)
        else:
            return self.base_model(x)
    
