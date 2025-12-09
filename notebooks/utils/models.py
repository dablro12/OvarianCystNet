# notebooks/utils/models.py
import torch
import torch.nn as nn
from torchvision import models
import math 
import timm   # ← ViT backbone 사용
# Attention MIL 모델
class AttentionMIL(nn.Module):
    """
    Ilse et al. (2018) 스타일 Attention-based MIL.
    - 입력: 한 Bag의 이미지들 (N_inst, C, H, W)
    - 출력: Bag-level logits (1, num_classes)
    """

    def __init__(self, num_classes: int, embed_dim: int = 256, attn_dim: int = 128):
        super().__init__()
        
        self.encoder = InstanceEncoder(embed_dim = embed_dim) # Instance Encoder 사용
        
        self.attention = nn.Sequential( # Attention 네트워크 => 여러 인스터스 중 중요한 인스턴스 선택
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1) 
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes) # Bag-Level Classifier
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batch dimension: if x is (1, N_inst, C, H, W), squeeze to (N_inst, C, H, W)
        if x.dim() == 5:
            x = x.squeeze(0)  # (1, N_inst, C, H, W) -> (N_inst, C, H, W)
        
        # 1. Instance Embedding
        h = self.encoder(x) # (N, embed_dim)
        
        # 2. Attention Score -> Weight
        att_raw = self.attention(h).squeeze(-1) # (N,)
        att_weight = torch.softmax(att_raw, dim=0) # (N,) # att_weight 는 각 인스턴스에 대한 가중치
        
        # 3. weighted sum(하나의 백 내 weighted sum은 1로 수렴) => bag embedding
        bag_emb = torch.sum(att_weight.unsqueeze(-1) * h, dim = 0) # (embed_dim,)
        
        # bag-level logits
        logits = self.classifier(bag_emb) # (num_classes,)
        logits = logits.unsqueeze(0)       # (num_classes,) -> (1, num_classes)
        
        return logits
    
# # 기본 MIL 모델
# model = AttentionMIL(num_classes = num_classes, embed_dim = 256, attn_dim = 128).to(device)

# Instance Encoder(CNN -> Feature)
class InstanceEncoder(nn.Module):
    """
    단일 ultrasound frame → feature embedding
    timm의 pretrained ViT를 feature extractor로 사용
    """
    def __init__(self, embed_dim=256, backbone="vit_base_patch16_224"):
        super().__init__()

        # ViT backbone (feature_dim 출력됨)
        # num_classes=0 → classification head 제거
        self.vit = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0
        )

        vit_dim = self.vit.num_features  # 보통 768

        # 최종 feature projection layer
        self.fc = nn.Linear(vit_dim, embed_dim)

    def forward(self, x):
        """
        x: (N_inst, C, H, W)
        return: (N_inst, embed_dim)
        """
        feat = self.vit(x)     # (N_inst, vit_dim)
        feat = self.fc(feat)   # (N_inst, embed_dim)
        return feat
    
# Positional Encoding(Bag 내 instance 위치정보를 추가해주기)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]
    
    
# Transformer Block 구현
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

class TransformerMIL(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_dim=256,
        depth=4,
        n_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        vit_backbone="vit_base_patch16_224",   # ★ 여기서 ViT backbone 선택
    ):
        super().__init__()

        # Instance Encoder → ViT 기반
        self.encoder = InstanceEncoder(
            embed_dim=embed_dim,
            backbone=vit_backbone
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, bag_imgs):
        # bag_imgs: (N_inst, C, H, W) or (1, N_inst, C, H, W)
        if bag_imgs.dim() == 5:
            bag_imgs = bag_imgs.squeeze(0)

        # 1) ViT instance features
        h = self.encoder(bag_imgs)  # (N_inst, D)

        h = h.unsqueeze(0)  # (1, N_inst, D)

        # 2) prepend CLS
        cls = self.cls_token
        x = torch.cat([cls, h], dim=1)

        # 3) positional encoding
        x = self.pos_encoder(x)

        # 4) Transformer encoder
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_rep = x[:, 0, :]  # (1, D)

        logits = self.fc(cls_rep)
        return logits