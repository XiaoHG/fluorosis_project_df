"""在线增强管线, 按 05_Exp_Design/02_augmentation.md 实现."""

from torchvision.transforms import (
    Compose, Resize, Normalize, RandomHorizontalFlip, RandomApply,
    RandomRotation, RandomResizedCrop, ColorJitter, GaussianBlur,
    RandomAdjustSharpness, ToTensor,
)
from torchvision.transforms.v2 import CutMix


def get_train_transform(image_size=(256, 512)):
    """训练集: resize → 几何→颜色→纹理 → ToTensor → Normalize."""
    return Compose([
        Resize(image_size),
        RandomHorizontalFlip(p=0.5),
        RandomApply([RandomRotation(degrees=10)], p=0.5),
        RandomApply([RandomResizedCrop(size=image_size, scale=(0.8, 1.0))], p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        RandomApply([RandomAdjustSharpness(sharpness_factor=1.5)], p=0.3),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size=(256, 512)):
    """验证集: resize → ToTensor → 归一化."""
    return Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_cutmix(alpha=0.5, num_classes=4):
    """batch 级 CutMix."""
    return CutMix(alpha=alpha, num_classes=num_classes)
