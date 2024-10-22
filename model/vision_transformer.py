import torch.nn as nn
from torchvision import models

class binary_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/vision_transformer.html
    """
    
    def __init__(self, type, num_classes = 1):
        super(binary_model, self).__init__()
        if type == 'b_16': # Resize : 224
            self.base_model = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif type == 'l_16': # Resize : 224
            self.base_model = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif type == 'h_14': # Resize : 480
            self.base_model = models.vit_h_14(weights = models.ViT_H_14_Weights.IMAGENET1K_V1)
            
        self.base_model.heads[-1] = nn.Linear(self.base_model.heads[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x).view(-1)

class multi_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/vision_transformer.html
    """
    
    def __init__(self, type, num_classes = 3):
        super(multi_model, self).__init__()
        if type == 'b_16': # Resize : 224
            self.base_model = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif type == 'l_16': # Resize : 224
            self.base_model = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif type == 'h_14': # Resize : 480
            self.base_model = models.vit_h_14(weights = models.ViT_H_14_Weights.IMAGENET1K_V1)
            
        self.base_model.heads[-1] = nn.Linear(self.base_model.heads[-1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)
    