
import timm
import torch.nn as nn 
from torchvision import models
class model_setup(nn.Module):
    """
        ref : https://pytorch.org/vision/stable/models/swin_transformer.html 
    """
    def __init__(self, type, num_classes):
        super(model_setup, self).__init__()
        self.num_classes = num_classes
        if type=="timm":
            self.base_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes = num_classes)
        else:
            if type =='t': # 256
                self.base_model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.IMAGENET1K_V1)
            elif type == 's': # 224
                self.base_model = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True,  num_classes=num_classes)
            elif type == 'default': # 256
                self.base_model = models.swin_v2_b(weights = models.Swin_V2_B_Weights.IMAGENET1K_V1)
            
            self.base_model.head = nn.Linear(self.base_model.head.in_features, num_classes)
        
    def forward(self, x):
        if self.num_classes == 1:
            return self.base_model(x).view(-1)
        else:
            return self.base_model(x)
    