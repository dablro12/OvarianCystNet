import torch.nn as nn
from torchvision import models

class binary_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
    """
    
    def __init__(self, type):
        super(binary_model, self).__init__()
        if type == 's': # Resize : 384
            self.base_model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights)
        elif type == 'm': # Resize : 480
            self.base_model = models.efficientnet_v2_m(weights = models.EfficientNet_V2_M_Weights)
        elif type == 'l': # Resize : 480
            self.base_model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights)
        self.base_model.classifier[-1] = nn.Linear(1280, 1)
            
        
    def forward(self, x):
        return self.base_model(x).view(-1)


class multi_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
    """
    
    def __init__(self, type):
        super(multi_model, self).__init__()
        if type == 's': # Resize : 384
            self.base_model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights)
        elif type == 'm': # Resize : 480
            self.base_model = models.efficientnet_v2_m(weights = models.EfficientNet_V2_M_Weights)
        elif type == 'l': # Resize : 480
            self.base_model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights)
            
        self.base_model.classifier[-1] = nn.Linear(1280, 3)
        
    def forward(self, x):
        return self.base_model(x)
    