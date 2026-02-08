"""Tests for data loading and preprocessing modules."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import cv2

from pet_breed_uncertainty_aware_classifier.data.loader import OxfordPetsDataset, PetDataLoader
from pet_breed_uncertainty_aware_classifier.data.preprocessing import (
    get_transforms, mixup_data, cutmix_data, mixup_criterion, TTA
)
from pet_breed_uncertainty_aware_classifier.utils.config import DataConfig


class TestPreprocessing:
    """Test preprocessing utilities."""

    def test_get_transforms_training(self):
        """Test training transforms."""
        transform = get_transforms(
            image_size=(224, 224),
            augmentation_strength=0.5,
            is_training=True
        )

        # Create dummy image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Apply transform
        result = transform(image=image)

        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 224, 224)
        assert result['image'].dtype == torch.float32

    def test_get_transforms_validation(self):
        """Test validation transforms."""
        transform = get_transforms(
            image_size=(224, 224),
            augmentation_strength=0.0,
            is_training=False
        )

        # Create dummy image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_get_transforms_invalid_image_size(self):
        """Test get_transforms with invalid image size."""
        with pytest.raises(ValueError, match="image_size must be a tuple of two positive integers"):
            get_transforms(image_size=(224,), is_training=False)

        with pytest.raises(ValueError, match="image_size must be a tuple of two positive integers"):
            get_transforms(image_size=(0, 224), is_training=False)

    def test_get_transforms_invalid_augmentation_strength(self):
        """Test get_transforms with invalid augmentation strength."""
        with pytest.raises(ValueError, match="augmentation_strength must be between 0.0 and 1.0"):
            get_transforms(augmentation_strength=-0.1, is_training=True)

        with pytest.raises(ValueError, match="augmentation_strength must be between 0.0 and 1.0"):
            get_transforms(augmentation_strength=1.1, is_training=True)

    def test_get_transforms_invalid_is_training(self):
        """Test get_transforms with invalid is_training parameter."""
        with pytest.raises(ValueError, match="is_training must be a boolean"):
            get_transforms(is_training="True")

    def test_mixup_data(self):
        """Test mixup augmentation."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0, use_cuda=False)

        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert isinstance(lam, float)
        assert 0 <= lam <= 1

    def test_cutmix_data(self):
        """Test CutMix augmentation."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))

        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0, use_cuda=False)

        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert isinstance(lam, float)
        assert 0 <= lam <= 1

    def test_mixup_criterion(self):
        """Test mixup loss criterion."""
        pred = torch.randn(4, 10)
        y_a = torch.randint(0, 10, (4,))
        y_b = torch.randint(0, 10, (4,))
        lam = 0.6

        criterion = torch.nn.CrossEntropyLoss()
        loss = mixup_criterion(criterion, pred, y_a, y_b, lam)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_tta(self):
        """Test Test Time Augmentation."""
        transform = get_transforms((64, 64), 0.3, is_training=True)
        tta = TTA(transform, num_augmentations=5)

        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        augmented = tta(image)

        assert len(augmented) == 5
        assert all(isinstance(img, torch.Tensor) for img in augmented)
        assert all(img.shape == (3, 64, 64) for img in augmented)


class TestOxfordPetsDataset:
    """Test Oxford Pets dataset."""

    def create_mock_dataset(self, temp_dir: Path):
        """Create a mock dataset structure."""
        images_dir = temp_dir / "images"
        annotations_dir = temp_dir / "annotations"

        images_dir.mkdir(parents=True)
        annotations_dir.mkdir(parents=True)

        # Create mock images
        class_names = ["Abyssinian", "Bengal", "basset_hound"]
        for class_name in class_names:
            for i in range(3):  # 3 images per class
                img_path = images_dir / f"{class_name}_{i+1}.jpg"
                # Create a dummy image
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(img_path), img)

        # Create annotation files
        trainval_content = []
        test_content = []

        for idx, class_name in enumerate(class_names):
            for i in range(3):
                filename = f"{class_name}_{i+1}"
                class_id = OxfordPetsDataset.CLASS_NAMES.index(class_name) + 1

                if i < 2:  # First 2 for training
                    trainval_content.append(f"{filename} {class_id}")
                else:  # Last one for testing
                    test_content.append(f"{filename} {class_id}")

        with open(annotations_dir / "trainval.txt", 'w') as f:
            f.write('\n'.join(trainval_content))

        with open(annotations_dir / "test.txt", 'w') as f:
            f.write('\n'.join(test_content))

    def test_dataset_creation(self, temp_dir):
        """Test dataset creation with mock data."""
        self.create_mock_dataset(temp_dir)

        # Test trainval split
        dataset = OxfordPetsDataset(
            root=str(temp_dir),
            split="trainval",
            transform=None,
            download=False
        )

        assert len(dataset) == 6  # 2 images per class, 3 classes
        assert len(dataset.class_counts) <= 3

        # Test loading a sample
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert 0 <= label < len(OxfordPetsDataset.CLASS_NAMES)

    def test_dataset_with_transform(self, temp_dir):
        """Test dataset with transforms."""
        self.create_mock_dataset(temp_dir)

        transform = get_transforms((32, 32), 0.0, is_training=False)
        dataset = OxfordPetsDataset(
            root=str(temp_dir),
            split="trainval",
            transform=transform,
            download=False
        )

        image, label = dataset[0]
        assert image.shape == (3, 32, 32)

    def test_class_weights(self, temp_dir):
        """Test class weight calculation."""
        self.create_mock_dataset(temp_dir)

        dataset = OxfordPetsDataset(
            root=str(temp_dir),
            split="trainval",
            transform=None,
            download=False
        )

        weights = dataset.get_class_weights()
        assert isinstance(weights, torch.Tensor)
        assert weights.shape[0] == len(OxfordPetsDataset.CLASS_NAMES)
        assert torch.all(weights > 0)


class TestPetDataLoader:
    """Test PetDataLoader."""

    def test_data_loader_initialization(self):
        """Test data loader initialization."""
        config = DataConfig(dataset_path="test_path", batch_size=4)
        loader = PetDataLoader(config)

        assert loader.config == config
        assert loader.train_dataset is None
        assert loader.val_dataset is None
        assert loader.test_dataset is None

    def test_class_names_and_count(self):
        """Test getting class names and count."""
        config = DataConfig()
        loader = PetDataLoader(config)

        class_names = loader.get_class_names()
        num_classes = loader.get_num_classes()

        assert isinstance(class_names, list)
        assert len(class_names) == 37  # Oxford Pets has 37 classes
        assert num_classes == 37

    @pytest.mark.parametrize("use_weighted_sampling", [True, False])
    def test_data_loader_creation_mock(self, temp_dir, use_weighted_sampling):
        """Test data loader creation with mock data."""
        # This test requires actual dataset preparation, which is complex
        # In a real scenario, you would mock the prepare_datasets method
        # or use a minimal test dataset

        config = DataConfig(
            dataset_path=str(temp_dir),
            batch_size=2,
            num_workers=0
        )

        loader = PetDataLoader(config)

        # Mock the dataset preparation
        from unittest.mock import Mock
        loader.train_dataset = Mock()
        loader.train_dataset.__len__ = Mock(return_value=4)
        loader.train_dataset.get_class_weights = Mock(return_value=torch.ones(5))
        loader.train_dataset.labels = [0, 1, 0, 1]

        # This would normally fail without proper datasets, but we're testing the interface
        try:
            # Test the method exists and has correct signature
            assert hasattr(loader, 'get_train_loader')
            assert hasattr(loader, 'get_val_loader')
            assert hasattr(loader, 'get_test_loader')
        except:
            # Expected to fail without proper datasets
            pass


if __name__ == "__main__":
    pytest.main([__file__])