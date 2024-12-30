from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms

# Grad-CAM을 위한 함수 정의
def apply_gradcam(model, images, target_layer =None):
    """
    모델과 이미지, 타겟 레이어를 받아 Grad-CAM 히트맵을 생성합니다.
    
    Parameters:
    - model (torch.nn.Module): 학습된 모델
    - images (torch.Tensor): 입력 이미지 텐서 (배치 크기 1)
    - target_layer (str): Grad-CAM을 적용할 타겟 레이어 이름
    
    Returns:
    - heatmap (numpy.ndarray): Grad-CAM 히트맵
    """
    cam_extractor = GradCAM(model)
    out = model(images)
    # 클래스가 1개인 경우 인덱스 0 사용
    target_class = out.argmax(dim=1).item()
    activation_map = cam_extractor(target_class, out)
    heatmap = activation_map.squeeze().cpu().numpy()
    return heatmap

def get_layer(model, model_name):
    """
    모델 이름에 따라 타겟 레이어를 반환합니다.
    """
    layer_path = MODEL_LAYERS.get(model_name, None)
    if not layer_path:
        raise ValueError(f"Unknown model type: {model_name}")
    
    attrs = layer_path.split('.')
    target_layer = model
    for attr in attrs:
        if '[' in attr and ']' in attr:
            # 예: 'layer4[-1]'
            attr_name, index = attr.split('[')
            index = int(index[:-1])  # 'layer4[-1]' -> 'layer4', '-1'
            target_layer = getattr(target_layer, attr_name)[index]
        elif attr.isdigit():
            # 인덱스가 숫자인 경우
            target_layer = target_layer[int(attr)]
        else:
            target_layer = getattr(target_layer, attr)
    return target_layer
