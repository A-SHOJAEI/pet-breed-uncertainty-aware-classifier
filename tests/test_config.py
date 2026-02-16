"""Tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path

from pet_breed_uncertainty_aware_classifier.utils.config import (
    Config, ModelConfig, DataConfig, TrainingConfig, LoggingConfig,
    load_config, save_config, _validate_model_config, _validate_data_config,
    _validate_training_config, _validate_logging_config, _validate_global_config
)


class TestConfigValidation:
    """Test configuration validation functions."""

    def test_validate_model_config_valid(self):
        """Test model config validation with valid parameters."""
        config = ModelConfig(
            backbone="efficientnet_b0",
            num_classes=37,
            dropout_rate=0.3,
            mc_samples=100,
            ensemble_size=5
        )
        # Should not raise any exception
        _validate_model_config(config)

    def test_validate_model_config_invalid_num_classes(self):
        """Test model config validation with invalid num_classes."""
        config = ModelConfig(num_classes=0)
        with pytest.raises(ValueError, match="num_classes must be positive"):
            _validate_model_config(config)

        config = ModelConfig(num_classes=-1)
        with pytest.raises(ValueError, match="num_classes must be positive"):
            _validate_model_config(config)

    def test_validate_model_config_invalid_dropout_rate(self):
        """Test model config validation with invalid dropout_rate."""
        config = ModelConfig(dropout_rate=-0.1)
        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            _validate_model_config(config)

        config = ModelConfig(dropout_rate=1.1)
        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 1.0"):
            _validate_model_config(config)

    def test_validate_model_config_invalid_mc_samples(self):
        """Test model config validation with invalid mc_samples."""
        config = ModelConfig(mc_samples=0)
        with pytest.raises(ValueError, match="mc_samples must be positive"):
            _validate_model_config(config)

    def test_validate_model_config_invalid_ensemble_size(self):
        """Test model config validation with invalid ensemble_size."""
        config = ModelConfig(ensemble_size=0)
        with pytest.raises(ValueError, match="ensemble_size must be positive"):
            _validate_model_config(config)

    def test_validate_data_config_valid(self):
        """Test data config validation with valid parameters."""
        config = DataConfig(
            batch_size=32,
            num_workers=4,
            image_size=(224, 224),
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            augmentation_strength=0.5
        )
        # Should not raise any exception
        _validate_data_config(config)

    def test_validate_data_config_invalid_batch_size(self):
        """Test data config validation with invalid batch_size."""
        config = DataConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            _validate_data_config(config)

    def test_validate_data_config_invalid_num_workers(self):
        """Test data config validation with invalid num_workers."""
        config = DataConfig(num_workers=-1)
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            _validate_data_config(config)

    def test_validate_data_config_invalid_image_size(self):
        """Test data config validation with invalid image_size."""
        config = DataConfig(image_size=(224,))
        with pytest.raises(ValueError, match="image_size must be a tuple of two positive integers"):
            _validate_data_config(config)

        config = DataConfig(image_size=(0, 224))
        with pytest.raises(ValueError, match="image_size must be a tuple of two positive integers"):
            _validate_data_config(config)

    def test_validate_data_config_invalid_splits(self):
        """Test data config validation with invalid split ratios."""
        config = DataConfig(train_split=0.0)
        with pytest.raises(ValueError, match="All split ratios must be between 0.0 and 1.0"):
            _validate_data_config(config)

        config = DataConfig(train_split=0.5, val_split=0.3, test_split=0.3)  # Sum > 1.0
        with pytest.raises(ValueError, match="Train, validation, and test splits must sum to 1.0"):
            _validate_data_config(config)

    def test_validate_data_config_invalid_augmentation_strength(self):
        """Test data config validation with invalid augmentation_strength."""
        config = DataConfig(augmentation_strength=-0.1)
        with pytest.raises(ValueError, match="augmentation_strength must be between 0.0 and 1.0"):
            _validate_data_config(config)

    def test_validate_training_config_valid(self):
        """Test training config validation with valid parameters."""
        config = TrainingConfig(
            epochs=50,
            learning_rate=1e-3,
            weight_decay=1e-4,
            scheduler="cosine",
            warmup_epochs=5,
            early_stopping_patience=10,
            gradient_clip_norm=1.0,
            label_smoothing=0.1
        )
        # Should not raise any exception
        _validate_training_config(config)

    def test_validate_training_config_invalid_epochs(self):
        """Test training config validation with invalid epochs."""
        config = TrainingConfig(epochs=0)
        with pytest.raises(ValueError, match="epochs must be positive"):
            _validate_training_config(config)

    def test_validate_training_config_invalid_learning_rate(self):
        """Test training config validation with invalid learning_rate."""
        config = TrainingConfig(learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            _validate_training_config(config)

    def test_validate_training_config_invalid_scheduler(self):
        """Test training config validation with invalid scheduler."""
        config = TrainingConfig(scheduler="invalid")
        with pytest.raises(ValueError, match="scheduler must be one of"):
            _validate_training_config(config)

    def test_validate_training_config_invalid_gradient_clip_norm(self):
        """Test training config validation with invalid gradient_clip_norm."""
        config = TrainingConfig(gradient_clip_norm=0)
        with pytest.raises(ValueError, match="gradient_clip_norm must be positive if specified"):
            _validate_training_config(config)

    def test_validate_logging_config_valid(self):
        """Test logging config validation with valid parameters."""
        config = LoggingConfig(
            log_level="INFO",
            log_interval=10,
            save_top_k=3,
            monitor_mode="max"
        )
        # Should not raise any exception
        _validate_logging_config(config)

    def test_validate_logging_config_invalid_log_level(self):
        """Test logging config validation with invalid log_level."""
        config = LoggingConfig(log_level="INVALID")
        with pytest.raises(ValueError, match="log_level must be one of"):
            _validate_logging_config(config)

    def test_validate_global_config_valid(self):
        """Test global config validation with valid parameters."""
        config = Config(seed=42, device="auto")
        config.training.epochs = 50
        config.training.warmup_epochs = 5
        # Should not raise any exception
        _validate_global_config(config)

    def test_validate_global_config_invalid_seed(self):
        """Test global config validation with invalid seed."""
        config = Config(seed=-1)
        with pytest.raises(ValueError, match="seed must be non-negative"):
            _validate_global_config(config)

    def test_validate_global_config_invalid_device(self):
        """Test global config validation with invalid device."""
        config = Config(device="invalid")
        with pytest.raises(ValueError, match="device must be one of"):
            _validate_global_config(config)

    def test_validate_global_config_invalid_warmup_epochs(self):
        """Test global config validation with warmup_epochs >= epochs."""
        config = Config()
        config.training.epochs = 10
        config.training.warmup_epochs = 10
        with pytest.raises(ValueError, match="warmup_epochs must be less than total epochs"):
            _validate_global_config(config)


class TestConfigLoadSave:
    """Test configuration loading and saving."""

    def test_load_config_valid_file(self, tmp_path):
        """Test loading a valid configuration file."""
        config_path = tmp_path / "config.yaml"

        config_data = {
            "model": {"backbone": "resnet50", "num_classes": 10},
            "data": {"batch_size": 16},
            "seed": 123
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))

        assert config.model.backbone == "resnet50"
        assert config.model.num_classes == 10
        assert config.data.batch_size == 16
        assert config.seed == 123

    def test_load_config_file_not_found(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("non_existent.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading a malformed YAML file."""
        config_path = tmp_path / "config.yaml"

        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError, match="Error parsing configuration file"):
            load_config(str(config_path))

    def test_load_config_empty_file(self, tmp_path):
        """Test loading an empty configuration file."""
        config_path = tmp_path / "config.yaml"
        config_path.touch()  # Create empty file

        config = load_config(str(config_path))

        # Should return default config
        assert isinstance(config, Config)
        assert config.model.backbone == "efficientnet_b0"

    def test_load_config_invalid_values(self, tmp_path):
        """Test loading configuration with invalid values."""
        config_path = tmp_path / "config.yaml"

        config_data = {
            "model": {"num_classes": -1}  # Invalid value
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            load_config(str(config_path))

    def test_save_config_valid(self, tmp_path):
        """Test saving a valid configuration."""
        config = Config()
        config.model.backbone = "resnet50"
        config.data.batch_size = 16

        config_path = tmp_path / "saved_config.yaml"
        save_config(config, str(config_path))

        assert config_path.exists()

        # Load and verify
        loaded_config = load_config(str(config_path))
        assert loaded_config.model.backbone == "resnet50"
        assert loaded_config.data.batch_size == 16

    def test_save_config_invalid_directory(self):
        """Test saving configuration to invalid directory."""
        config = Config()
        invalid_path = "/invalid/path/config.yaml"

        with pytest.raises(IOError, match="Unable to save configuration"):
            save_config(config, invalid_path)

    def test_save_config_invalid_config(self, tmp_path):
        """Test saving invalid configuration."""
        config = Config()
        config.seed = -1  # Invalid value

        config_path = tmp_path / "config.yaml"

        with pytest.raises(ValueError, match="seed must be non-negative"):
            save_config(config, str(config_path))