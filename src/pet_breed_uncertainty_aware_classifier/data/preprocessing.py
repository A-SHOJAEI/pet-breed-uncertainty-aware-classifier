"""Data preprocessing and augmentation utilities."""

import logging
from typing import Callable, Tuple, Optional, Union
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2


logger = logging.getLogger(__name__)


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_strength: float = 0.5,
    is_training: bool = True
) -> Callable:
    """
    Get image transformations for training or validation.

    Args:
        image_size: Target image size (height, width)
        augmentation_strength: Strength of augmentations (0.0 to 1.0)
        is_training: Whether to apply training augmentations

    Returns:
        Albumentations transform function

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if len(image_size) != 2 or any(s <= 0 for s in image_size):
        raise ValueError("image_size must be a tuple of two positive integers")

    if not 0.0 <= augmentation_strength <= 1.0:
        raise ValueError("augmentation_strength must be between 0.0 and 1.0")

    if not isinstance(is_training, bool):
        raise ValueError("is_training must be a boolean")
    base_transforms = [
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            always_apply=True
        ),
        ToTensorV2()
    ]

    if not is_training:
        return A.Compose(base_transforms)

    # Training augmentations with strength scaling
    train_transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=image_size,
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            p=0.8
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1 * augmentation_strength,
            scale_limit=0.2 * augmentation_strength,
            rotate_limit=int(15 * augmentation_strength),
            border_mode=0,
            p=0.7
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0 * augmentation_strength, 50.0 * augmentation_strength)),
            A.GaussianBlur(blur_limit=(1, 3)),
            A.MotionBlur(blur_limit=3),
        ], p=0.4),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2 * augmentation_strength),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2 * augmentation_strength,
                contrast_limit=0.2 * augmentation_strength
            ),
        ], p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=int(20 * augmentation_strength),
            sat_shift_limit=int(30 * augmentation_strength),
            val_shift_limit=int(20 * augmentation_strength),
            p=0.4
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=int(image_size[0] * 0.1 * augmentation_strength),
            max_width=int(image_size[1] * 0.1 * augmentation_strength),
            min_holes=1,
            fill_value=0,
            p=0.3
        ),
    ]

    # Add base transforms at the end
    train_transforms.extend([
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            always_apply=True
        ),
        ToTensorV2()
    ])

    return A.Compose(train_transforms)


def mixup_data(
    x: Tensor,
    y: Tensor,
    alpha: float = 1.0,
    use_cuda: bool = True
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    Apply mixup augmentation to batch data.

    Args:
        x: Input batch of shape (batch_size, channels, height, width)
        y: Target labels of shape (batch_size,)
        alpha: Mixup interpolation strength
        use_cuda: Whether to use CUDA for random number generation

    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda) where:
        - mixed_x: Mixed input data
        - y_a, y_b: Original target pairs
        - lambda: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: Tensor,
    y: Tensor,
    alpha: float = 1.0,
    use_cuda: bool = True
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    Apply CutMix augmentation to batch data.

    Args:
        x: Input batch of shape (batch_size, channels, height, width)
        y: Target labels of shape (batch_size,)
        alpha: CutMix interpolation strength
        use_cuda: Whether to use CUDA for random number generation

    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda) where:
        - mixed_x: Mixed input data with rectangular patches
        - y_a, y_b: Original target pairs
        - lambda: Mixing coefficient based on area ratio
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # Get image dimensions
    _, _, H, W = x.shape

    # Generate random bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling of center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to exactly match the area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(
    criterion: Callable,
    pred: Tensor,
    y_a: Tensor,
    y_b: Tensor,
    lam: float
) -> Tensor:
    """
    Compute mixup loss combining two target labels.

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of target labels
        y_b: Second set of target labels
        lam: Mixing coefficient

    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class TTA(object):
    """Test Time Augmentation for improved uncertainty estimation."""

    def __init__(self, transforms: Callable, num_augmentations: int = 8):
        """
        Initialize TTA.

        Args:
            transforms: Augmentation transforms to apply
            num_augmentations: Number of augmented versions per image
        """
        self.transforms = transforms
        self.num_augmentations = num_augmentations

    def __call__(self, image: np.ndarray) -> list:
        """
        Apply multiple augmentations to input image.

        Args:
            image: Input image as numpy array

        Returns:
            List of augmented images as tensors
        """
        augmented_images = []
        for _ in range(self.num_augmentations):
            augmented = self.transforms(image=image)['image']
            augmented_images.append(augmented)
        return augmented_images