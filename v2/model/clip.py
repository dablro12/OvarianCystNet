import torch
import torch.nn as nn 
from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import CLIPModel, CLIPProcessor, CLIPConfig

model_history = {
    'clipbase' : 'openai/clip-vit-base-patch32',
    'clipgmp' : 'zer0int/CLIP-GmP-ViT-L-14',
    "biomedclip" : 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    'MedCSPclip' : 'hf-hub:xcwangpsu/MedCSP_clip',
}
class model_setup(nn.Module):
    def __init__(self, type, num_class=1):
        super(model_setup, self).__init__()
        self.num_classes = num_class
        try:
            self.clip_model, self.preprocess = create_model_from_pretrained(
                model_history[type.split('-')[0]],
                precision = 'fp32',
                device='cuda'
            )
            self.tokenizer = get_tokenizer(
                model_history[type.split('-')[0]]
            )
        except:
            try: # for tranformers 4.35.2 for python 3.10
                model_id = model_history[type.split('-')[0]]
                config = CLIPConfig.from_pretrained(model_id)
                self.clip_model = CLIPModel.from_pretrained(model_id, config=config)
                self.preprocess = CLIPProcessor.from_pretrained(model_id)
            except:
                print('CLIP model is not loaded')
            
        
        # 모델을 float32로 강제 고정
        self.clip_model = self.clip_model.float()  
        self.logit_scale = self.clip_model.logit_scale.exp()

        if type.split('-')[1] == 'linearprob':
            self.classifier = torch.nn.Linear(512, num_class).cuda()
        elif type.split('-')[1] == 'CoCoOP':
            pass
        else:
            pass

        for param in self.clip_model.parameters(): # clip model은 파라미터 고정 classifier만 학습
            param.requires_grad = True
        print(f"[CLIP Weight] 🧊 Freeze")

    def forward(self, x):
        # 입력도 float32로 맞춘다
        x = x.to('cuda:0').to(torch.float32)

        image_features = self.clip_model.encode_image(x)
        logit = self.classifier(image_features)
        if self.num_classes == 1:
            return logit.view(-1)
        else:
            return logit

