import torch
import torch.nn as nn
from torchvision import models

class model_setup(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/vision_transformer.html
    """
    
    def __init__(self, type, num_classes):
        super(model_setup, self).__init__()
        self.num_classes = num_classes
        if type == "timm":
            self.base_model = torch.hub.load('facebookresearch/deit:main','deit_tiny_patch16_224', pretrained=True)
            self.base_model.head = nn.Linear(self.base_model.head.in_features, num_classes)
        else:
            if type == 'b_16': # Resize : 224
                self.base_model = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
            elif type == 'l_16': # Resize : 224
                self.base_model = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_V1)
            elif type == 'h_14': # Resize : 480
                self.base_model = models.vit_h_14(weights = models.ViT_H_14_Weights.IMAGENET1K_V1)
            self.base_model.heads[-1] = nn.Linear(self.base_model.heads[-1].in_features, num_classes)
        
    def forward(self, x):
        if self.num_classes == 1:
            return self.base_model(x).view(-1)
        else:
            return self.base_model(x)

