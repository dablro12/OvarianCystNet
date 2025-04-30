# DinoV2 모델에 분류 레이어 추가
import torch.nn as nn
import torch 
class DinoClassifier(nn.Module):
    def __init__(self, model_name, num_classes, freeze_backbone=False):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        
        # 백본 모델 가중치 고정 (freeze)
        if freeze_backbone:
            print("[INFO] Backbone 가중치 Freeze")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 특성 차원 (1024)에서 분류 레이어 추가
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.linear_head.out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        # 특성 벡터의 [CLS] 토큰 사용
        return self.classifier(features)