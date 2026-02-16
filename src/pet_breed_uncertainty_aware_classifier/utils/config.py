"""Configuration management for the pet breed classifier."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    backbone: str = "efficientnet_b0"
    num_classes: int = 37
    dropout_rate: float = 0.3
    mc_samples: int = 100
    ensemble_size: int = 5
    pretrained: bool = True
    freeze_backbone: bool = False


@dataclass
class DataConfig:
    """Data configuration parameters."""

    dataset_path: str = "data/oxford-iiit-pet"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    augmentation_strength: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip_norm: Optional[float] = 1.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""

    log_level: str = "INFO"
    mlflow_tracking_uri: str = "mlruns"
    experiment_name: str = "pet_breed_uncertainty"
    log_interval: int = 10
    save_top_k: int = 3
    monitor: str = "val_accuracy"
    monitor_mode: str = "max"


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Runtime configurations
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    compile_model: bool = False


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Loaded configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
        ValueError: If configuration values are invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            logger.warning(f"Empty configuration file: {config_path}, using defaults")
            return Config()

        # Convert nested dict to dataclass instances with validation
        config = Config()

        if "model" in config_dict:
            try:
                config.model = ModelConfig(**config_dict["model"])
                _validate_model_config(config.model)
            except TypeError as e:
                raise ValueError(f"Invalid model configuration: {e}")

        if "data" in config_dict:
            try:
                data_config = config_dict["data"].copy()
                # Convert image_size list back to tuple if needed
                if 'image_size' in data_config and isinstance(data_config['image_size'], list):
                    data_config['image_size'] = tuple(data_config['image_size'])
                config.data = DataConfig(**data_config)
                _validate_data_config(config.data)
            except TypeError as e:
                raise ValueError(f"Invalid data configuration: {e}")

        if "training" in config_dict:
            try:
                config.training = TrainingConfig(**config_dict["training"])
                _validate_training_config(config.training)
            except TypeError as e:
                raise ValueError(f"Invalid training configuration: {e}")

        if "logging" in config_dict:
            try:
                config.logging = LoggingConfig(**config_dict["logging"])
                _validate_logging_config(config.logging)
            except TypeError as e:
                raise ValueError(f"Invalid logging configuration: {e}")

        # Set top-level attributes with validation
        for key, value in config_dict.items():
            if hasattr(config, key) and key not in ["model", "data", "training", "logging"]:
                try:
                    setattr(config, key, value)
                except Exception as e:
                    raise ValueError(f"Invalid value for {key}: {e}")

        _validate_global_config(config)
        logger.info(f"Configuration loaded from {config_path}")
        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, yaml.YAMLError, ValueError)):
            raise
        else:
            raise ValueError(f"Unexpected error loading configuration: {e}")


def _validate_model_config(model_config: ModelConfig) -> None:
    """
    Validate model configuration parameters.

    Args:
        model_config: Model configuration to validate

    Raises:
        ValueError: If configuration values are invalid
    """
    if model_config.num_classes <= 0:
        raise ValueError("num_classes must be positive")

    if not 0.0 <= model_config.dropout_rate <= 1.0:
        raise ValueError("dropout_rate must be between 0.0 and 1.0")

    if model_config.mc_samples <= 0:
        raise ValueError("mc_samples must be positive")

    if model_config.ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive")


def _validate_data_config(data_config: DataConfig) -> None:
    """
    Validate data configuration parameters.

    Args:
        data_config: Data configuration to validate

    Raises:
        ValueError: If configuration values are invalid
    """
    if data_config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if data_config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    if len(data_config.image_size) != 2 or any(s <= 0 for s in data_config.image_size):
        raise ValueError("image_size must be a tuple of two positive integers")

    # Validate split ratios
    splits = [data_config.train_split, data_config.val_split, data_config.test_split]
    if not all(0.0 < s < 1.0 for s in splits):
        raise ValueError("All split ratios must be between 0.0 and 1.0")

    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test splits must sum to 1.0")

    if not 0.0 <= data_config.augmentation_strength <= 1.0:
        raise ValueError("augmentation_strength must be between 0.0 and 1.0")


def _validate_training_config(training_config: TrainingConfig) -> None:
    """
    Validate training configuration parameters.

    Args:
        training_config: Training configuration to validate

    Raises:
        ValueError: If configuration values are invalid
    """
    if training_config.epochs <= 0:
        raise ValueError("epochs must be positive")

    if training_config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if training_config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    valid_schedulers = ["cosine", "cosine_warm", "step", "none"]
    if training_config.scheduler not in valid_schedulers:
        raise ValueError(f"scheduler must be one of {valid_schedulers}")

    if training_config.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")

    if training_config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")

    if (training_config.gradient_clip_norm is not None and
        training_config.gradient_clip_norm <= 0):
        raise ValueError("gradient_clip_norm must be positive if specified")

    if not 0.0 <= training_config.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be between 0.0 and 1.0 (exclusive)")


def _validate_logging_config(logging_config: LoggingConfig) -> None:
    """
    Validate logging configuration parameters.

    Args:
        logging_config: Logging configuration to validate

    Raises:
        ValueError: If configuration values are invalid
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if logging_config.log_level not in valid_levels:
        raise ValueError(f"log_level must be one of {valid_levels}")

    if logging_config.log_interval <= 0:
        raise ValueError("log_interval must be positive")

    if logging_config.save_top_k <= 0:
        raise ValueError("save_top_k must be positive")

    valid_modes = ["min", "max"]
    if logging_config.monitor_mode not in valid_modes:
        raise ValueError(f"monitor_mode must be one of {valid_modes}")


def _validate_global_config(config: Config) -> None:
    """
    Validate global configuration parameters.

    Args:
        config: Complete configuration to validate

    Raises:
        ValueError: If configuration values are invalid
    """
    if config.seed < 0:
        raise ValueError("seed must be non-negative")

    valid_devices = ["auto", "cpu", "cuda", "mps"]
    if config.device not in valid_devices:
        raise ValueError(f"device must be one of {valid_devices}")

    # Cross-validation checks
    if config.training.warmup_epochs >= config.training.epochs:
        raise ValueError("warmup_epochs must be less than total epochs")


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        config_path: Path where to save the configuration file

    Raises:
        IOError: If unable to write to the specified path
        ValueError: If configuration is invalid
    """
    try:
        config_file = Path(config_path)

        # Validate configuration before saving
        _validate_global_config(config)

        # Create parent directories if they don't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dicts, handling special types
        model_dict = config.model.__dict__.copy()
        data_dict = config.data.__dict__.copy()
        training_dict = config.training.__dict__.copy()
        logging_dict = config.logging.__dict__.copy()

        # Convert tuples to lists for YAML serialization
        if 'image_size' in data_dict and isinstance(data_dict['image_size'], tuple):
            data_dict['image_size'] = list(data_dict['image_size'])

        config_dict = {
            "model": model_dict,
            "data": data_dict,
            "training": training_dict,
            "logging": logging_dict,
            "seed": config.seed,
            "device": config.device,
            "mixed_precision": config.mixed_precision,
            "compile_model": config.compile_model,
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {config_path}")

    except (OSError, IOError) as e:
        raise IOError(f"Unable to save configuration to {config_path}: {e}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error serializing configuration: {e}")