"""Data loading utilities for Oxford Pets dataset."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets.utils import download_and_extract_archive
import cv2
from sklearn.model_selection import train_test_split

from ..utils.config import DataConfig


logger = logging.getLogger(__name__)


class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pet Dataset with uncertainty-aware features."""

    CLASS_NAMES = [
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau',
        'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx',
        'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
        'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter',
        'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
        'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian',
        'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
        'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
    ]

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        """
        Initialize Oxford Pets dataset.

        Args:
            root: Root directory for dataset
            split: Dataset split ('trainval', 'test', or custom list)
            transform: Image transformations
            download: Whether to download dataset if not present
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.download_flag = download

        self._prepare_dataset()
        self._load_annotations()

    def _prepare_dataset(self) -> None:
        """Download and prepare the dataset."""
        self.root.mkdir(parents=True, exist_ok=True)

        images_dir = self.root / "images"
        annotations_dir = self.root / "annotations"

        if not images_dir.exists() or not annotations_dir.exists():
            if self.download_flag:
                logger.info("Downloading Oxford-IIIT Pet Dataset...")
                self._download_dataset()
            else:
                raise RuntimeError(
                    f"Dataset not found at {self.root}. Set download=True to download."
                )

    def _download_dataset(self) -> None:
        """Download the Oxford Pets dataset."""
        base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"

        # Download images
        try:
            download_and_extract_archive(
                f"{base_url}images.tar.gz",
                self.root,
                filename="images.tar.gz"
            )

            # Download annotations
            download_and_extract_archive(
                f"{base_url}annotations.tar.gz",
                self.root,
                filename="annotations.tar.gz"
            )

            logger.info("Dataset download completed successfully.")

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def _load_annotations(self) -> None:
        """Load dataset annotations and create splits."""
        annotations_dir = self.root / "annotations"

        # Load class mappings
        self.images = []
        self.labels = []
        self.class_counts = {}

        # Parse annotation files
        trainval_file = annotations_dir / "trainval.txt"
        test_file = annotations_dir / "test.txt"

        if not trainval_file.exists() or not test_file.exists():
            # Create annotations from image files
            self._create_annotations_from_images()
        else:
            # Load existing annotations
            if self.split == "trainval":
                self._load_split_file(trainval_file)
            elif self.split == "test":
                self._load_split_file(test_file)
            else:
                # Custom split provided as list
                self.images = self.split
                self.labels = self._extract_labels_from_filenames(self.images)

        # Calculate class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique, counts))

        logger.info(f"Loaded {len(self.images)} images from {self.split} split")
        logger.info(f"Class distribution: {self.class_counts}")

    def _create_annotations_from_images(self) -> None:
        """Create annotations from image filenames when annotation files are missing."""
        images_dir = self.root / "images"

        all_images = []
        all_labels = []

        for img_path in images_dir.glob("*.jpg"):
            # Extract class name from filename
            filename = img_path.stem
            class_name = "_".join(filename.split("_")[:-1])

            if class_name in self.CLASS_NAMES:
                all_images.append(str(img_path))
                all_labels.append(self.CLASS_NAMES.index(class_name))

        # Create train/test split
        if len(all_images) == 0:
            raise RuntimeError("No valid images found in dataset directory")

        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )

        if self.split in ["trainval", "train"]:
            self.images = train_imgs
            self.labels = train_labels
        else:  # test split
            self.images = test_imgs
            self.labels = test_labels

    def _load_split_file(self, split_file: Path) -> None:
        """Load images and labels from split file."""
        images_dir = self.root / "images"

        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = int(parts[1]) - 1  # Convert to 0-based indexing
                    img_path = images_dir / f"{img_name}.jpg"

                    if img_path.exists():
                        self.images.append(str(img_path))
                        self.labels.append(class_id)

    def _extract_labels_from_filenames(self, image_paths: List[str]) -> List[int]:
        """Extract class labels from image filenames."""
        labels = []
        for img_path in image_paths:
            filename = Path(img_path).stem
            class_name = "_".join(filename.split("_")[:-1])

            if class_name in self.CLASS_NAMES:
                labels.append(self.CLASS_NAMES.index(class_name))
            else:
                logger.warning(f"Unknown class name: {class_name}")
                labels.append(0)  # Default to first class

        return labels

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.

        Returns:
            Tensor of class weights
        """
        if not self.class_counts:
            return torch.ones(len(self.CLASS_NAMES))

        total_samples = sum(self.class_counts.values())
        weights = []

        for class_id in range(len(self.CLASS_NAMES)):
            count = self.class_counts.get(class_id, 1)
            weight = total_samples / (len(self.CLASS_NAMES) * count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


class PetDataLoader:
    """Data loader manager for Oxford Pets dataset."""

    def __init__(self, config: DataConfig):
        """
        Initialize data loader.

        Args:
            config: Data configuration
        """
        self.config = config
        self.train_dataset: Optional[OxfordPetsDataset] = None
        self.val_dataset: Optional[OxfordPetsDataset] = None
        self.test_dataset: Optional[OxfordPetsDataset] = None

    def prepare_datasets(
        self,
        train_transform: Callable,
        val_transform: Callable
    ) -> None:
        """
        Prepare train, validation, and test datasets.

        Args:
            train_transform: Training transformations
            val_transform: Validation/test transformations
        """
        from .preprocessing import get_transforms

        # Load full training set
        full_dataset = OxfordPetsDataset(
            root=self.config.dataset_path,
            split="trainval",
            transform=None,
            download=True
        )

        # Create train/validation split
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            full_dataset.images,
            full_dataset.labels,
            test_size=self.config.val_split / (self.config.train_split + self.config.val_split),
            stratify=full_dataset.labels,
            random_state=42
        )

        # Create dataset instances
        self.train_dataset = OxfordPetsDataset(
            root=self.config.dataset_path,
            split=train_imgs,
            transform=train_transform,
            download=False
        )

        self.val_dataset = OxfordPetsDataset(
            root=self.config.dataset_path,
            split=val_imgs,
            transform=val_transform,
            download=False
        )

        self.test_dataset = OxfordPetsDataset(
            root=self.config.dataset_path,
            split="test",
            transform=val_transform,
            download=False
        )

        logger.info(f"Training set: {len(self.train_dataset)} samples")
        logger.info(f"Validation set: {len(self.val_dataset)} samples")
        logger.info(f"Test set: {len(self.test_dataset)} samples")

    def get_train_loader(self, use_weighted_sampling: bool = True) -> DataLoader:
        """
        Create training data loader.

        Args:
            use_weighted_sampling: Whether to use weighted random sampling

        Returns:
            Training data loader
        """
        if self.train_dataset is None:
            raise RuntimeError("Datasets not prepared. Call prepare_datasets() first.")

        sampler = None
        if use_weighted_sampling:
            class_weights = self.train_dataset.get_class_weights()
            sample_weights = [class_weights[label] for label in self.train_dataset.labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )

    def get_val_loader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Datasets not prepared. Call prepare_datasets() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def get_test_loader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Datasets not prepared. Call prepare_datasets() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return OxfordPetsDataset.CLASS_NAMES

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(OxfordPetsDataset.CLASS_NAMES)