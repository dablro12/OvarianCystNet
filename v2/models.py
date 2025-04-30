from torchvision import models
import torch.nn as nn
import os 
import timm
# Load Model weight
def get_checkpoint_path(checkpoint_dir, datetime):
    ckpt_paths = [os.path.join(checkpoint_dir, f'{datetime}_fold{num}.pth')for num in range(1,6)]
    # 파일이 있는지 확인
    for ckpt_path in ckpt_paths:
        if os.path.isfile(ckpt_path):
            # drop
            pass
        else:
            print(f"{ckpt_path} is Not found")
            ckpt_paths.remove(ckpt_path)
    
    return ckpt_paths

def Model_Loader(model_name, num_classes):
    #%% [Model] ResNet Series
    if model_name == 'resnet_18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet_34':
        model = models.resnet34(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet_50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet_101':
        model = models.resnet101(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet_152':
        model = models.resnet152(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    

    #%% [Model] ConvNeXt Series
    elif model_name =='convnext_tiny': # 224
        model = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'convnext_s': # 224
        model = models.convnext_small(weights = models.ConvNeXt_Small_Weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'convnext_b': # 224
        model = models.convnext_base(weights = models.ConvNeXt_Base_Weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'convnext_l': # 224
        model = models.convnext_large(weights = models.ConvNeXt_Large_Weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    #%% [Model] Vision Transformer Series
    elif model_name == 'vit_b_16': # Resize : 224
        model = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
    elif model_name == 'vit_l_16': # Resize : 224
        model = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_V1)
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
    elif model_name == 'vit_h_14': # Resize : 480
        model = models.vit_h_14(weights = models.ViT_H_14_Weights.IMAGENET1K_V1)
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
    
    #%% [Model] VGG Series
    elif model_name == 'vit_vgg16': # 224
        model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    elif model_name =='vit_vgg19': # 224
        model = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(4096, num_classes)

    #%% [Model] Swin Transformer Series
    elif model_name =='swin_t': # 256
        model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'swin_s': # 224
        model = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True,  num_classes=num_classes)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'swin_default': # 256
        model = models.swin_v2_b(weights = models.Swin_V2_B_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)

    #%% [Model] MobileNet Series
    elif model_name == 'mobilenet_s': # 224
        model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(1024, num_classes)
    elif model_name == 'mobilenet_l': # 224
        model = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        model.classifier[-1] = nn.Linear(1280, num_classes)

    #%% [Model] Max Axial Vision Transform Series
    elif model_name == 'maxvit_t': # 224
        model = models.maxvit_t(weights = models.MaxVit_T_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    #%% [Model] EfficientNet Series
    elif model_name == 'efficientnet_s': # Resize : 384
        model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights)
        model.classifier[-1] = nn.Linear(1280, num_classes)
    elif model_name == 'efficientnet_m': # Resize : 480
        model = models.efficientnet_v2_m(weights = models.EfficientNet_V2_M_Weights)
        model.classifier[-1] = nn.Linear(1280, num_classes)
    elif model_name == 'efficientnet_l': # Resize : 480
        model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights)
        model.classifier[-1] = nn.Linear(1280, num_classes)
    
    #%% [Model] CycleGAN Series
    elif model_name == "CycleGAN":
        from model.cycleGAN import Gen, Dis
        model = dict()
        model['G'] = Gen
        model['D'] = Dis
    
    #%% [Model] DinoV2 Series
    elif model_name.split('_')[0] == "dinov2":
        from model.dino_v2 import DinoClassifier
        model = DinoClassifier(model_name = model_name, num_classes=num_classes)

    else:
        raise ValueError("Model Name is not Correct")
    
    
        
    return model