"""Tests for training modules."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from pet_breed_uncertainty_aware_classifier.training.trainer import (
    UncertaintyAwareLoss, EarlyStopping, UncertaintyTrainer
)
from pet_breed_uncertainty_aware_classifier.models.model import UncertaintyAwareClassifier
from pet_breed_uncertainty_aware_classifier.evaluation.metrics import UncertaintyMetrics


class TestUncertaintyAwareLoss:
    """Test uncertainty-aware loss function."""

    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = UncertaintyAwareLoss(
            base_loss="cross_entropy",
            label_smoothing=0.1,
            uncertainty_weight=0.1,
            diversity_weight=0.01
        )

        assert isinstance(loss_fn.base_criterion, nn.CrossEntropyLoss)
        assert loss_fn.uncertainty_weight == 0.1
        assert loss_fn.diversity_weight == 0.01

    def test_focal_loss_initialization(self):
        """Test focal loss initialization."""
        loss_fn = UncertaintyAwareLoss(base_loss="focal")
        assert callable(loss_fn.base_criterion)

    def test_single_model_loss(self):
        """Test loss computation for single model."""
        loss_fn = UncertaintyAwareLoss()

        outputs = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        losses = loss_fn(outputs, targets)

        assert 'classification' in losses
        assert 'uncertainty' in losses
        assert 'diversity' in losses
        assert 'total' in losses

        assert isinstance(losses['total'], torch.Tensor)
        assert losses['total'].dim() == 0  # Scalar

    def test_ensemble_loss(self):
        """Test loss computation for ensemble."""
        loss_fn = UncertaintyAwareLoss(diversity_weight=0.1)

        # Simulate ensemble outputs
        outputs = [torch.randn(4, 10) for _ in range(3)]
        targets = torch.randint(0, 10, (4,))

        losses = loss_fn(outputs, targets)

        assert 'classification' in losses
        assert 'diversity' in losses
        assert 'total' in losses

        # Diversity loss should be non-zero for ensemble
        assert losses['diversity'].item() != 0.0

    def test_focal_loss_computation(self):
        """Test focal loss computation."""
        loss_fn = UncertaintyAwareLoss(base_loss="focal")

        outputs = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = loss_fn._focal_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_diversity_loss(self):
        """Test diversity loss computation."""
        loss_fn = UncertaintyAwareLoss()

        outputs = [torch.randn(4, 10) for _ in range(3)]
        diversity_loss = loss_fn._compute_diversity_loss(outputs)

        assert isinstance(diversity_loss, torch.Tensor)
        # Diversity loss should be negative (encouraging diversity)
        assert diversity_loss.item() <= 0


class TestEarlyStopping:
    """Test early stopping utility."""

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        early_stopping = EarlyStopping(patience=3, mode='max')

        # Should not stop on first call
        assert not early_stopping(0.5)

        # Should not stop on improvement
        assert not early_stopping(0.7)

        # Should not stop on small decline within patience
        assert not early_stopping(0.6)  # 1st non-improvement
        assert not early_stopping(0.65) # 2nd non-improvement

        # Should stop after patience exceeded
        assert early_stopping(0.64)     # 3rd non-improvement, triggers stopping

    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode."""
        early_stopping = EarlyStopping(patience=2, mode='min')

        # Should not stop on first call
        assert not early_stopping(1.0)

        # Should not stop on improvement
        assert not early_stopping(0.8)

        # Should not stop within patience
        assert not early_stopping(0.9)

        # Should stop after patience exceeded
        assert early_stopping(0.95)

    def test_early_stopping_min_delta(self):
        """Test early stopping with minimum delta."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='max')

        assert not early_stopping(0.5)
        # Small improvement below min_delta should not reset counter
        assert not early_stopping(0.505)
        # Should stop after patience
        assert early_stopping(0.506)


class TestUncertaintyTrainer:
    """Test uncertainty trainer."""

    def test_trainer_initialization(self, sample_model, sample_config):
        """Test trainer initialization."""
        device = torch.device("cpu")  # Force CPU for consistent testing
        with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
            trainer = UncertaintyTrainer(sample_model, sample_config, device=device)

        assert trainer.model == sample_model
        assert trainer.config == sample_config
        assert trainer.device == device
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert isinstance(trainer.criterion, UncertaintyAwareLoss)
        assert isinstance(trainer.early_stopping, EarlyStopping)

    def test_device_setup(self, sample_model, sample_config):
        """Test device setup."""
        # Test auto device selection
        sample_config.device = "auto"
        with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
            trainer = UncertaintyTrainer(sample_model, sample_config)

        assert trainer.device in [torch.device("cpu"), torch.device("cuda"), torch.device("mps")]

        # Test explicit CPU device
        sample_config.device = "cpu"
        with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
            trainer = UncertaintyTrainer(sample_model, sample_config)

        assert trainer.device == torch.device("cpu")

    @patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow')
    def test_train_epoch_basic(self, mock_mlflow, sample_model, sample_config, sample_batch):
        """Test basic training epoch."""
        trainer = UncertaintyTrainer(sample_model, sample_config)

        # Create a minimal data loader
        from torch.utils.data import DataLoader, TensorDataset

        data, targets = sample_batch
        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Run training epoch
        metrics = trainer.train_epoch(dataloader, epoch=1)

        # Check that metrics are returned
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert isinstance(metrics['loss'], float)
        assert isinstance(metrics['accuracy'], float)
        assert 0 <= metrics['accuracy'] <= 100

    @patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow')
    def test_validate(self, mock_mlflow, sample_model, sample_config, sample_batch):
        """Test validation."""
        trainer = UncertaintyTrainer(sample_model, sample_config)

        # Create a minimal data loader
        from torch.utils.data import DataLoader, TensorDataset

        data, targets = sample_batch
        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Run validation
        metrics = trainer.validate(dataloader)

        # Check that comprehensive metrics are returned
        expected_keys = [
            'loss', 'accuracy', 'expected_calibration_error',
            'maximum_calibration_error', 'brier_score'
        ]
        assert all(key in metrics for key in expected_keys)

    def test_checkpoint_save_load(self, sample_model, sample_config, temp_dir):
        """Test checkpoint saving and loading."""
        with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
            trainer = UncertaintyTrainer(sample_model, sample_config)

        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        trainer._save_checkpoint(checkpoint_path, epoch=5, metrics={'accuracy': 0.8})

        assert checkpoint_path.exists()

        # Load checkpoint
        with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
            new_trainer = UncertaintyTrainer(sample_model, sample_config)

        checkpoint_data = new_trainer.load_checkpoint(str(checkpoint_path))

        assert checkpoint_data['epoch'] == 5
        assert checkpoint_data['metrics']['accuracy'] == 0.8
        assert new_trainer.current_epoch == 5

    @patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow')
    def test_mixed_loss_computation(self, mock_mlflow, sample_model, sample_config):
        """Test mixed loss computation for mixup/cutmix."""
        trainer = UncertaintyTrainer(sample_model, sample_config)

        # Create mock outputs and targets
        outputs = torch.randn(4, sample_config.model.num_classes)
        targets_a = torch.randint(0, sample_config.model.num_classes, (4,))
        targets_b = torch.randint(0, sample_config.model.num_classes, (4,))
        lam = 0.6

        loss_dict = trainer._compute_mixed_loss(outputs, targets_a, targets_b, lam)

        assert 'classification' in loss_dict
        assert 'uncertainty' in loss_dict
        assert 'diversity' in loss_dict
        assert 'total' in loss_dict

        assert isinstance(loss_dict['total'], torch.Tensor)
        assert loss_dict['total'].dim() == 0

    def test_scheduler_setup(self, sample_model, sample_config):
        """Test different scheduler setups."""
        schedulers_to_test = ["cosine", "cosine_warm", "step"]

        for scheduler_name in schedulers_to_test:
            sample_config.training.scheduler = scheduler_name

            with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
                trainer = UncertaintyTrainer(sample_model, sample_config)

            assert trainer.scheduler is not None

            # Test scheduler step
            initial_lr = trainer.optimizer.param_groups[0]['lr']
            trainer.scheduler.step()

            # Learning rate might change depending on scheduler
            final_lr = trainer.optimizer.param_groups[0]['lr']
            assert isinstance(final_lr, float)

    def test_mixed_precision_setup(self, sample_model, sample_config):
        """Test mixed precision training setup."""
        sample_config.mixed_precision = True

        with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
            trainer = UncertaintyTrainer(sample_model, sample_config)

        assert trainer.scaler is not None

        sample_config.mixed_precision = False

        with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
            trainer = UncertaintyTrainer(sample_model, sample_config)

        assert trainer.scaler is None


class TestTrainingUtils:
    """Test training utility functions and integration."""

    def test_gradient_clipping_values(self):
        """Test different gradient clipping values."""
        model = nn.Linear(10, 5)

        # Create some gradients
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Test gradient clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

        # Apply clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check that gradients exist
        assert any(p.grad is not None for p in model.parameters())

    def test_training_state_tracking(self, sample_model, sample_config):
        """Test training state tracking."""
        with patch('pet_breed_uncertainty_aware_classifier.training.trainer.mlflow'):
            trainer = UncertaintyTrainer(sample_model, sample_config)

        # Check initial state
        assert trainer.current_epoch == 0
        assert len(trainer.training_history['train_loss']) == 0

        # Simulate adding to history
        trainer.training_history['train_loss'].append(0.5)
        trainer.training_history['val_loss'].append(0.4)

        assert len(trainer.training_history['train_loss']) == 1
        assert trainer.training_history['train_loss'][0] == 0.5


if __name__ == "__main__":
    pytest.main([__file__])