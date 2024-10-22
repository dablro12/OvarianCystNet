import torchvision.models.alexnet as alexnet 
import torch.nn as nn 
class binary_model(nn.Module):
    """ ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html"""
    def __init__(self, type):
        super(binary_model, self).__init__()
        self.base_model = alexnet(weights='IMAGENET1K_V1')
        self.base_model.classifier[-1] = nn.Linear(4096, 3)
        
    def forward(self, x):
        out = self.base_model(x).view(-1)
        return out
    
class multi_model(nn.Module):
    """ ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html"""
    def __init__(self, type):
        super(multi_model, self).__init__()
        self.base_model = alexnet(weights='IMAGENET1K_V1')
        self.base_model.classifier[-1] = nn.Linear(4096, 3)
        
    def forward(self, x):
        return self.base_model(x)
        
