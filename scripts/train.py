#!/usr/bin/env python3
"""Training script for pet breed uncertainty-aware classifier."""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pet_breed_uncertainty_aware_classifier.utils.config import load_config, Config
from pet_breed_uncertainty_aware_classifier.data.loader import PetDataLoader
from pet_breed_uncertainty_aware_classifier.data.preprocessing import get_transforms
from pet_breed_uncertainty_aware_classifier.models.model import UncertaintyAwareClassifier
from pet_breed_uncertainty_aware_classifier.training.trainer import UncertaintyTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility, but may impact performance
    cudnn.deterministic = True
    cudnn.benchmark = False


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train uncertainty-aware pet breed classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actual training (for testing setup)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting pet breed uncertainty-aware classifier training")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Load configuration
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)

        config = load_config(args.config)

        # Override device if specified
        if args.device != "auto":
            config.device = args.device

        logger.info(f"Configuration loaded successfully")
        logger.info(f"Model: {config.model.backbone}")
        logger.info(f"Ensemble size: {config.model.ensemble_size}")
        logger.info(f"Training device: {config.device}")

        # Set random seed
        set_seed(config.seed)
        logger.info(f"Random seed set to {config.seed}")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup data loaders
        logger.info("Setting up data loaders...")
        data_loader = PetDataLoader(config.data)

        # Get transforms
        train_transform = get_transforms(
            image_size=config.data.image_size,
            augmentation_strength=config.data.augmentation_strength,
            is_training=True
        )
        val_transform = get_transforms(
            image_size=config.data.image_size,
            augmentation_strength=0.0,
            is_training=False
        )

        # Prepare datasets
        data_loader.prepare_datasets(train_transform, val_transform)

        # Create data loaders
        train_loader = data_loader.get_train_loader(use_weighted_sampling=True)
        val_loader = data_loader.get_val_loader()

        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Number of classes: {data_loader.get_num_classes()}")

        # Create model
        logger.info("Creating uncertainty-aware model...")
        model = UncertaintyAwareClassifier(config.model)

        # Log model info
        model_info = model.get_model_info()
        logger.info(f"Model architecture: {model_info['model_type']}")
        logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")

        # Create trainer
        logger.info("Setting up trainer...")
        trainer = UncertaintyTrainer(model, config)

        # Resume from checkpoint if specified
        if args.resume:
            if Path(args.resume).exists():
                logger.info(f"Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
            else:
                logger.error(f"Checkpoint not found: {args.resume}")
                sys.exit(1)

        if args.dry_run:
            logger.info("Dry run completed successfully. Exiting.")
            return

        # Start training
        logger.info("Starting training...")
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=output_dir / "checkpoints"
        )

        # Save training history
        history_path = output_dir / "training_history.json"
        import json
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, values in training_history.items():
                if isinstance(values, list):
                    serializable_history[key] = [
                        float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for v in values
                    ]
                else:
                    serializable_history[key] = values
            json.dump(serializable_history, f, indent=2)

        logger.info(f"Training history saved to {history_path}")

        # Final evaluation on validation set
        logger.info("Running final evaluation...")
        final_metrics = trainer.validate(val_loader)

        logger.info("Final Validation Metrics:")
        logger.info(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        logger.info(f"  Expected Calibration Error: {final_metrics['expected_calibration_error']:.4f}")
        logger.info(f"  Maximum Calibration Error: {final_metrics['maximum_calibration_error']:.4f}")
        logger.info(f"  Brier Score: {final_metrics['brier_score']:.4f}")

        # Save final configuration
        from pet_breed_uncertainty_aware_classifier.utils.config import save_config
        config_save_path = output_dir / "final_config.yaml"
        save_config(config, str(config_save_path))
        logger.info(f"Configuration saved to {config_save_path}")

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()