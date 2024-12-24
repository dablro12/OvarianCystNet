import torch.nn as nn
from torchvision import models
from transformers import ViTImageProcessor, ViTForImageClassification

class binary_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/vision_transformer.html
    """
    def __init__(self, type, num_classes=1):
        super(binary_model, self).__init__()
        self.base_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.base_model.classifier = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    # def forward(self, x):
    #     # base_model의 출력 중 logits만 반환
    #     outputs = self.base_model(x)
    #     return outputs.logits.view(-1)  # binary classification에 맞게 변경
    def forward(self, x):
        # logits만 반환
        return self.base_model(x).logits.view(-1)

class multi_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/vision_transformer.html
    """
    def __init__(self, type, num_classes=3):
        super(multi_model, self).__init__()
        self.base_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.base_model.classifier = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    # def forward(self, x):
    #     # base_model의 출력 중 logits만 반환
    #     outputs = self.base_model(x)
    #     return outputs.logits  # multi-class classification에 맞게 logits 반환
    def forward(self, x):
        # logits만 반환
        return self.base_model(x).logits