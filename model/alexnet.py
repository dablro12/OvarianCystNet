import torchvision.models.alexnet as alexnet 
import torch.nn as nn 

""" ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html"""
class model_setup(nn.Module):
    def __init__(self, type, num_classes):
        super(model_setup, self).__init__()
        self.num_classes = num_classes
        self.base_model = alexnet(weights='IMAGENET1K_V1')
        self.base_model.classifier[-1] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        if self.num_classes == 1:
            return self.base_model(x).view(-1)
        else:
            return self.base_model(x)
    