from model.backbone.dynamic_vit import LVViTDiffPruning
import torch 
import torch.nn as nn 
PRUNING_LOC = [5, 10, 15]
BASE_RATE = 0.7
KEEP_RATE = [BASE_RATE, BASE_RATE ** 2, BASE_RATE ** 3]

class multi_model(nn.Module):
    def __init__(self):
        super(multi_model, self).__init__()
        self.base_model = LVViTDiffPruning(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True, mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, viz_mode=True,
        )
        
        checkpoint = torch.load("/mnt/hdd/octc/experiment/preweight/dynamic-vit_lv-s_r0.5.pth", map_location = 'cpu')['model']# Load Model Weights
        self.base_model.load_state_dict(checkpoint)
        
        self.base_model.classifier[-1] = nn.Linear(512, 3)
        
    def forward(self, x):
        return self.base_model(x)
    
    
    