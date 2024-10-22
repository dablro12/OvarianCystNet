import torch.nn as nn
from torchvision import models

class binary_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.maxvit_t.html#torchvision.models.maxvit_t
    """
    
    def __init__(self, type, num_classes= 1):
        super(binary_model, self).__init__()
        if type == 'default': # Resize : 224
            self.base_model = models.maxvit_t(weights = models.MaxVit_T_Weights.IMAGENET1K_V1)
            
        self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x).view(-1)

class multi_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.maxvit_t.html#torchvision.models.maxvit_t
    """
    
    def __init__(self, type, num_classes = 3):
        super(multi_model, self).__init__()
        if type == 'default': # Resize : 224
            self.base_model = models.maxvit_t(weights = models.MaxVit_T_Weights.IMAGENET1K_V1)
            
        self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)
    