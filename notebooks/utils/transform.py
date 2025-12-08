from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import random

class SpeckleNoise(object):
    """
    초음파 이미지 전용 speckle noise augmentation
    """
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor

    def __call__(self, img):
        noise = torch.randn_like(img) * self.noise_factor
        return img + img * noise
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.03):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def get_transform(train_transform=None):
    """
    Return train and val transforms with optional augmentation.
    
    Parameters
    ----------
    train_transform : list[torchvision.transforms]
        추가로 적용할 train 전용 transform 리스트.
    
    Returns
    -------
    train_tf : torchvision.transforms.Compose
    val_tf : torchvision.transforms.Compose
    """

    if train_transform is None:
        train_transform = []

    # Base transform (공통)
    base_resize = transforms.Resize((224, 224))
    base_to_tensor = transforms.ToTensor()
    base_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Train Transform
    train_tf = transforms.Compose([
        base_resize,
        *train_transform,   # augmentation
        base_to_tensor,
        base_norm,
    ])

    # Validation Transform (augmentation 없이 resize → tensor → normalize만)
    val_tf = transforms.Compose([
        base_resize,
        base_to_tensor,
        base_norm,
    ])

    return train_tf, val_tf
