
from segment_anything import sam_model_registry
import torch.nn as nn
import torch


# MLP Head 정의
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class binary_model(nn.Module):
    """ input size = 1024,1024 """
    def __init__(self, type, model_ckpt = '/home/eiden/eiden/PCOS-roi-classification/checkpoints/MedSAM1/medsam_vit_b.pth'):
        super(binary_model, self).__init__()
        medsam_backbone = sam_model_registry[type](checkpoint=model_ckpt)
        self.medsam_encoder = medsam_backbone.image_encoder

        # Freeze encoder
        for param in self.medsam_encoder.parameters():
            param.requires_grad = False
        print(f"[MEDSAM1] Encoder is frozen")
            
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),  # 채널 축소 (256 → 1)
            nn.Sigmoid()  # [0, 1]로 정규화
        )
        
        # Fully Connected Layers for Classification
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # Encoder 채널 수 256 → 중간 차원 128
            nn.ReLU(),
            nn.Dropout(0.5),  # 과적합 방지
            nn.Linear(128, 1)  # 최종 출력
        )
        
    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.medsam_encoder(x)  # [Batch, 256, 64, 64]
        
        # Spatial Attention
        attention = self.spatial_attention(x)  # [Batch, 1, 64, 64]
        
        # Element-wise 곱
        x = x * attention  # [Batch, 256, 64, 64]
        
        # Global Average Pooling
        x = torch.sum(x, dim=[2, 3])  # [Batch, 256]
        
        # Fully Connected Layers
        x = self.classifier(x)  # [Batch, n_classes]
        
        return x.view(-1)

class multi_model(nn.Module):
    """ input size = 1024,1024 """
    def __init__(self, type, model_ckpt = '/home/eiden/eiden/PCOS-roi-classification/checkpoints/MedSAM1/medsam_vit_b.pth'):
        super(multi_model, self).__init__()
        medsam_backbone = sam_model_registry[type](checkpoint=model_ckpt)
        self.medsam_encoder = medsam_backbone.image_encoder

        # Freeze encoder
        for param in self.medsam_encoder.parameters():
            param.requires_grad = True
        print(f"[MEDSAM1] Encoder is frozen")

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),  # 채널 축소 (256 → 1)
            nn.Sigmoid()  # [0, 1]로 정규화
        )
        
        # Fully Connected Layers for Classification
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # Encoder 채널 수 256 → 중간 차원 128
            nn.ReLU(),
            nn.Dropout(0.5),  # 과적합 방지
            nn.Linear(128, 3)  # 최종 출력
        )
        
    def forward(self, x):
        # Encoder를 통해 특징 추출
        x = self.medsam_encoder(x)  # [Batch, 256, 64, 64]
        
        # Spatial Attention
        attention = self.spatial_attention(x)  # [Batch, 1, 64, 64]
        
        # Element-wise 곱
        x = x * attention  # [Batch, 256, 64, 64]
        
        # Global Average Pooling
        x = torch.sum(x, dim=[2, 3])  # [Batch, 256]
        
        # Fully Connected Layers
        x = self.classifier(x)  # [Batch, n_classes]
        return x
        
    
# # Image Encoder에 MLP Head를 추가하는 클래스 정의
# class ImageEncoderWithMLPHead(nn.Module):
#     def __init__(self, image_encoder, hidden_dim, out_dim):
#         super(ImageEncoderWithMLPHead, self).__init__()
#         self.image_encoder = image_encoder
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 전역 평균 풀링
#         self.flatten = nn.Flatten()
#         self.mlp_head = MLPHead(in_dim=256, hidden_dim=hidden_dim, out_dim=out_dim)  # in_dim을 256으로 설정
        
#     def forward(self, x):
#         x = self.image_encoder(x)          # (batch_size, 256, H, W)
#         x = self.global_avg_pool(x)        # (batch_size, 256, 1, 1)
#         x = self.flatten(x)                # (batch_size, 256)
#         x = self.mlp_head(x)               # (batch_size, out_dim)
#         return x

# # MLP 헤드를 추가한 이미지 인코더 초기화
# hidden_dim = 512
# num_classes = 10
# image_encoder_with_mlp_head = ImageEncoderWithMLPHead(image_encoder, hidden_dim=hidden_dim, out_dim=num_classes)
# image_encoder_with_mlp_head = image_encoder_with_mlp_head.to(device)