"""Test configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from pet_breed_uncertainty_aware_classifier.utils.config import Config, ModelConfig, DataConfig, TrainingConfig
from pet_breed_uncertainty_aware_classifier.models.model import UncertaintyAwareClassifier


@pytest.fixture
def device():
    """Get test device (CPU for consistent testing)."""
    return torch.device("cpu")


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = Config()
    config.model = ModelConfig(
        backbone="efficientnet_b0",
        num_classes=5,  # Small number for testing
        dropout_rate=0.3,
        mc_samples=10,  # Small number for testing
        ensemble_size=2,  # Small ensemble for testing
        pretrained=False  # Avoid downloading in tests
    )
    config.data = DataConfig(
        dataset_path="test_data",
        image_size=(64, 64),  # Small images for testing
        batch_size=4,
        num_workers=0  # No multiprocessing in tests
    )
    config.training = TrainingConfig(
        epochs=2,  # Short training for testing
        learning_rate=1e-3
    )
    return config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_model(sample_config, device):
    """Create a sample model for testing."""
    model = UncertaintyAwareClassifier(sample_config.model)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_batch(device):
    """Create a sample batch of data for testing."""
    batch_size = 4
    channels = 3
    height = 64
    width = 64

    data = torch.randn(batch_size, channels, height, width, device=device)
    targets = torch.randint(0, 5, (batch_size,), device=device)

    return data, targets


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing metrics."""
    np.random.seed(42)

    num_samples = 100
    num_classes = 5

    # Generate logits
    logits = np.random.randn(num_samples, num_classes)

    # Convert to probabilities
    predictions = torch.softmax(torch.from_numpy(logits), dim=1)

    # Generate targets
    targets = torch.randint(0, num_classes, (num_samples,))

    # Generate uncertainties and confidences
    uncertainties = torch.rand(num_samples)
    confidences = torch.max(predictions, dim=1)[0]

    return {
        'predictions': predictions,
        'targets': targets,
        'uncertainties': uncertainties,
        'confidences': confidences
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False