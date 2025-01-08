import torch.nn as nn
from torchvision import models

class model_setup(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.maxvit_t.html#torchvision.models.maxvit_t
    """
    def __init__(self, type, num_classes):
        super(model_setup, self).__init__()
        self.num_classes = num_classes
        if type == 'default': # Resize : 224
            self.base_model = models.maxvit_t(weights = models.MaxVit_T_Weights.IMAGENET1K_V1)
            
        self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, num_classes)
        
    def forward(self, x):
        if self.num_classes == 1:
            return self.base_model(x).view(-1)
        else:
            return self.base_model(x)
