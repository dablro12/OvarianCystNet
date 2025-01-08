import torch.nn as nn
from torchvision import models
from transformers import ViTImageProcessor, ViTForImageClassification

class model_setup(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/vision_transformer.html
    """
    def __init__(self, type, num_classes):
        super(model_setup, self).__init__()
        self.num_classes = num_classes
        self.base_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.base_model.classifier = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    # def forward(self, x):
    #     # base_model의 출력 중 logits만 반환
    #     outputs = self.base_model(x)
    #     return outputs.logits.view(-1)  # binary classification에 맞게 변경
    def forward(self, x):
        # logits만 반환
        if self.num_classes == 1:
            return self.base_model(x).logits.view(-1)
        else:
            return self.base_model(x).logits
