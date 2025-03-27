import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import torchvision.transforms.functional as TF


class SpeckleNoise(object):
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            img_tensor = transforms.ToTensor()(img)
        noise = torch.randn_like(img_tensor) * self.noise_level
        noisy_img = img_tensor + img_tensor * noise
        noisy_img = torch.clamp(noisy_img, 0., 1.)
        # PIL 이미지로 변환하지 않고 텐서를 그대로 반환
        return noisy_img


class PairedAffineTransform:
    def __init__(self, degrees=15, translate=0.1, scale=(0.8, 1.2), shear=10):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img, mask):
        # 공통 파라미터 생성
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(-self.shear, self.shear)

        # 동일한 파라미터 적용
        img = TF.affine(
            img, angle, (tx, ty), scale, shear,
            interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.affine(
            mask, angle, (tx, ty), scale, shear,
            interpolation=TF.InterpolationMode.NEAREST  # 마스크는 Nearest 사용
        )
        return img, mask