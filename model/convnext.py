import torch.nn as nn
from torchvision import models

class model_setup(nn.Module):
    def __init__(self, type, num_classes):
        super(model_setup, self).__init__()
        self.num_classes = num_classes
        if type =='tiny': # 224
            self.base_model = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights)
        elif type == 's': # 224
            self.base_model = models.convnext_small(weights = models.ConvNeXt_Small_Weights)
        elif type == 'b': # 224
            self.base_model = models.convnext_base(weights = models.ConvNeXt_Base_Weights)
        elif type == 'l': # 224
            self.base_model = models.convnext_large(weights = models.ConvNeXt_Large_Weights)
        self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, num_classes)
    def forward(self, x):
        if self.num_classes == 1:
            return self.base_model(x).view(-1)
        else:
            return self.base_model(x)

