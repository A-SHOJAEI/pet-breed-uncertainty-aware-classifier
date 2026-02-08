"""Advanced trainer with uncertainty-aware training and MLflow tracking."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import mlflow
import mlflow.pytorch
import numpy as np
from tqdm import tqdm

from ..models.model import UncertaintyAwareClassifier
from ..evaluation.metrics import UncertaintyMetrics
from ..data.preprocessing import mixup_data, cutmix_data, mixup_criterion
from ..utils.config import Config


logger = logging.getLogger(__name__)


class UncertaintyAwareLoss(nn.Module):
    """Advanced loss function for uncertainty-aware training."""

    def __init__(
        self,
        base_loss: str = "cross_entropy",
        label_smoothing: float = 0.0,
        uncertainty_weight: float = 0.1,
        diversity_weight: float = 0.01
    ):
        """
        Initialize uncertainty-aware loss.

        Args:
            base_loss: Base loss function type
            label_smoothing: Label smoothing factor
            uncertainty_weight: Weight for uncertainty regularization
            diversity_weight: Weight for ensemble diversity loss
        """
        super().__init__()

        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight

        # Base classification loss
        if base_loss == "cross_entropy":
            self.base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif base_loss == "focal":
            self.base_criterion = self._focal_loss
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")

    def forward(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-aware loss.

        Args:
            outputs: Model outputs (single tensor or list for ensemble)
            targets: Target labels
            model: Model instance for accessing uncertainty predictions

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Handle ensemble vs single model outputs
        if isinstance(outputs, list):
            # Ensemble case
            individual_losses = []
            for output in outputs:
                loss = self.base_criterion(output, targets)
                individual_losses.append(loss)

            losses['classification'] = torch.mean(torch.stack(individual_losses))

            # Add diversity loss for ensemble
            if self.diversity_weight > 0:
                losses['diversity'] = self._compute_diversity_loss(outputs)
            else:
                losses['diversity'] = torch.tensor(0.0, device=outputs[0].device)

        else:
            # Single model case
            losses['classification'] = self.base_criterion(outputs, targets)
            losses['diversity'] = torch.tensor(0.0, device=outputs.device)

        # Determine device from outputs
        _device = outputs[0].device if isinstance(outputs, list) else outputs.device

        # Add uncertainty regularization if model supports it
        if model is not None and hasattr(model, 'predict_with_uncertainty'):
            if self.uncertainty_weight > 0:
                losses['uncertainty'] = self._compute_uncertainty_regularization(
                    model, outputs, targets
                )
            else:
                losses['uncertainty'] = torch.tensor(0.0, device=_device)
        else:
            losses['uncertainty'] = torch.tensor(0.0, device=_device)

        # Total loss
        total_loss = (losses['classification'] +
                     self.uncertainty_weight * losses['uncertainty'] +
                     self.diversity_weight * losses['diversity'])

        losses['total'] = total_loss
        return losses

    def _focal_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """Compute focal loss for addressing class imbalance."""
        ce_loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def _compute_diversity_loss(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute diversity loss for ensemble training."""
        if len(outputs) < 2:
            return torch.tensor(0.0, device=outputs[0].device)

        # Convert logits to probabilities
        probs = [torch.softmax(output, dim=1) for output in outputs]

        # Compute pairwise KL divergences
        diversity_loss = 0.0
        num_pairs = 0

        for i in range(len(probs)):
            for j in range(i + 1, len(probs)):
                # KL divergence between ensemble members
                kl_div = nn.functional.kl_div(
                    torch.log(probs[i] + 1e-8),
                    probs[j],
                    reduction='batchmean'
                )
                diversity_loss += kl_div
                num_pairs += 1

        return -diversity_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

    def _compute_uncertainty_regularization(
        self,
        model: nn.Module,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty regularization term."""
        # Determine the device from outputs (handle both tensor and list of tensors)
        if isinstance(outputs, list):
            device = outputs[0].device
        else:
            device = outputs.device

        try:
            # Skip uncertainty regularization for ensemble outputs (list)
            # as it would require running MC dropout on image inputs, not logits
            if isinstance(outputs, list):
                return torch.tensor(0.0, device=device)

            # Get uncertainty predictions
            with torch.no_grad():
                if hasattr(model, 'module'):  # Handle DataParallel
                    uncertainty_results = model.module.predict_with_uncertainty(
                        outputs.detach(), uncertainty_method="mc_dropout"
                    )
                else:
                    uncertainty_results = model.predict_with_uncertainty(
                        outputs.detach(), uncertainty_method="mc_dropout"
                    )

            uncertainties = uncertainty_results['total_uncertainty']

            # Encourage higher uncertainty for incorrect predictions
            predicted_classes = torch.argmax(outputs, dim=1)
            incorrect_mask = (predicted_classes != targets).float()

            # Regularization: encourage high uncertainty for wrong predictions
            uncertainty_reg = -torch.mean(uncertainties * incorrect_mask)

            return uncertainty_reg

        except Exception as e:
            logger.warning(f"Failed to compute uncertainty regularization: {e}")
            return torch.tensor(0.0, device=device)


class EarlyStopping:
    """Early stopping utility with model checkpointing."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for monitoring metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current monitoring metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class UncertaintyTrainer:
    """Advanced trainer for uncertainty-aware pet breed classification."""

    def __init__(
        self,
        model: UncertaintyAwareClassifier,
        config: Config,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Uncertainty-aware model
            config: Training configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device or self._setup_device()

        # Move model to device
        self.model.to(self.device)

        # Setup loss function
        self.criterion = UncertaintyAwareLoss(
            label_smoothing=config.training.label_smoothing,
            uncertainty_weight=0.1,
            diversity_weight=0.01 if config.model.ensemble_size > 1 else 0.0
        )

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup mixed precision if enabled
        self.scaler = GradScaler('cuda') if config.mixed_precision else None

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode=config.logging.monitor_mode
        )

        # Setup metrics
        self.uncertainty_metrics = UncertaintyMetrics()

        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf') if config.logging.monitor_mode == 'max' else float('inf')
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'val_ece': [], 'learning_rate': []
        }

        # Setup MLflow
        self._setup_mlflow()

        logger.info(f"Trainer initialized with device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        return device

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        if self.config.training.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.scheduler == "cosine_warm":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.config.training.learning_rate * 0.01
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )

    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        try:
            mlflow.set_tracking_uri(self.config.logging.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.logging.experiment_name)

            # Start MLflow run
            mlflow.start_run()

            # Log configuration
            mlflow.log_params({
                'model_backbone': self.config.model.backbone,
                'num_classes': self.config.model.num_classes,
                'ensemble_size': self.config.model.ensemble_size,
                'mc_samples': self.config.model.mc_samples,
                'dropout_rate': self.config.model.dropout_rate,
                'batch_size': self.config.data.batch_size,
                'learning_rate': self.config.training.learning_rate,
                'epochs': self.config.training.epochs,
                'weight_decay': self.config.training.weight_decay,
                'label_smoothing': self.config.training.label_smoothing,
                'mixup_alpha': self.config.training.mixup_alpha,
                'cutmix_alpha': self.config.training.cutmix_alpha,
            })

            logger.info("MLflow tracking initialized")

        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_uncertainty_loss = 0.0
        total_diversity_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)

            # Apply data augmentation (mixup/cutmix)
            if np.random.random() < 0.5 and self.config.training.mixup_alpha > 0:
                data, targets_a, targets_b, lam = mixup_data(
                    data, targets, self.config.training.mixup_alpha
                )
                mixed_targets = True
            elif np.random.random() < 0.5 and self.config.training.cutmix_alpha > 0:
                data, targets_a, targets_b, lam = cutmix_data(
                    data, targets, self.config.training.cutmix_alpha
                )
                mixed_targets = True
            else:
                mixed_targets = False

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(data)
                    if mixed_targets:
                        loss_dict = self._compute_mixed_loss(
                            outputs, targets_a, targets_b, lam
                        )
                    else:
                        loss_dict = self.criterion(outputs, targets, self.model)

                self.scaler.scale(loss_dict['total']).backward()

                if self.config.training.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                if mixed_targets:
                    loss_dict = self._compute_mixed_loss(outputs, targets_a, targets_b, lam)
                else:
                    loss_dict = self.criterion(outputs, targets, self.model)

                loss_dict['total'].backward()

                if self.config.training.gradient_clip_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )

                self.optimizer.step()

            # Update metrics
            total_loss += loss_dict['total'].item()
            total_classification_loss += loss_dict['classification'].item()
            total_uncertainty_loss += loss_dict['uncertainty'].item()
            total_diversity_loss += loss_dict['diversity'].item()

            # Calculate accuracy
            if not mixed_targets:
                if isinstance(outputs, list):
                    # Ensemble: average predictions
                    ensemble_output = torch.mean(torch.stack(outputs), dim=0)
                    predicted = ensemble_output.argmax(dim=1)
                else:
                    predicted = outputs.argmax(dim=1)
                correct_predictions += (predicted == targets).sum().item()

            total_samples += data.size(0)

            # Update progress bar
            if batch_idx % self.config.logging.log_interval == 0:
                progress_bar.set_postfix({
                    'Loss': f"{loss_dict['total'].item():.4f}",
                    'Acc': f"{100.0 * correct_predictions / total_samples:.2f}%"
                })

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        avg_uncertainty_loss = total_uncertainty_loss / len(train_loader)
        avg_diversity_loss = total_diversity_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples

        return {
            'loss': avg_loss,
            'classification_loss': avg_classification_loss,
            'uncertainty_loss': avg_uncertainty_loss,
            'diversity_loss': avg_diversity_loss,
            'accuracy': accuracy
        }

    def _compute_mixed_loss(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for mixed targets (mixup/cutmix)."""
        if isinstance(outputs, list):
            # Ensemble case
            mixed_losses = []
            for output in outputs:
                loss = mixup_criterion(
                    nn.functional.cross_entropy,
                    output, targets_a, targets_b, lam
                )
                mixed_losses.append(loss)

            classification_loss = torch.mean(torch.stack(mixed_losses))

            # Add diversity loss
            if self.criterion.diversity_weight > 0:
                diversity_loss = self.criterion._compute_diversity_loss(outputs)
            else:
                diversity_loss = torch.tensor(0.0, device=outputs[0].device)
        else:
            # Single model case
            classification_loss = mixup_criterion(
                nn.functional.cross_entropy,
                outputs, targets_a, targets_b, lam
            )
            diversity_loss = torch.tensor(0.0, device=outputs.device)

        # No uncertainty loss for mixed targets
        uncertainty_loss = torch.tensor(0.0)

        return {
            'classification': classification_loss,
            'uncertainty': uncertainty_loss,
            'diversity': diversity_loss,
            'total': classification_loss + self.criterion.diversity_weight * diversity_loss
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        all_predictions = []
        all_uncertainties = []
        all_confidences = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)

                # Get predictions with uncertainty
                uncertainty_results = self.model.predict_with_uncertainty(
                    data, uncertainty_method="combined"
                )

                predictions = uncertainty_results['predictions']
                uncertainties = uncertainty_results['total_uncertainty']
                confidences = uncertainty_results['confidence']

                # Calculate loss
                if isinstance(self.model.forward(data), list):
                    outputs = self.model.forward(data)
                    loss = torch.mean(torch.stack([
                        nn.functional.cross_entropy(output, targets)
                        for output in outputs
                    ]))
                else:
                    outputs = self.model.forward(data)
                    loss = nn.functional.cross_entropy(outputs, targets)

                total_loss += loss.item()

                # Collect predictions
                all_predictions.append(predictions.cpu())
                all_uncertainties.append(uncertainties.cpu())
                all_confidences.append(confidences.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute comprehensive metrics
        metrics = self.uncertainty_metrics.compute_all_metrics(
            predictions=all_predictions,
            targets=all_targets,
            uncertainties=all_uncertainties,
            confidences=all_confidences
        )

        # Add loss to metrics
        metrics['loss'] = total_loss / len(val_loader)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str = "checkpoints"
    ) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {self.config.training.epochs} epochs")

        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader, epoch + 1)

            # Validation
            val_metrics = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step()

            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_ece'].append(val_metrics['expected_calibration_error'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Log to MLflow
            try:
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_ece': val_metrics['expected_calibration_error'],
                    'val_mce': val_metrics['maximum_calibration_error'],
                    'val_brier_score': val_metrics['brier_score'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

            # Check for improvement
            monitor_metric = val_metrics[self.config.logging.monitor.replace('val_', '')]

            is_best = False
            if self.config.logging.monitor_mode == 'max':
                is_best = monitor_metric > self.best_metric
            else:
                is_best = monitor_metric < self.best_metric

            if is_best:
                self.best_metric = monitor_metric
                self._save_checkpoint(save_path / "best_model.pth", epoch, val_metrics)

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    save_path / f"checkpoint_epoch_{epoch+1}.pth",
                    epoch, val_metrics
                )

            # Early stopping
            if self.early_stopping(monitor_metric):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Log epoch summary
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.epochs} "
                f"({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Val ECE: {val_metrics['expected_calibration_error']:.4f}"
            )

        # Save final model
        self._save_checkpoint(save_path / "final_model.pth", epoch, val_metrics)

        # End MLflow run
        try:
            mlflow.end_run()
        except:
            pass

        logger.info("Training completed successfully")
        return self.training_history

    def _save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

        # Skip MLflow model logging (too slow for large models)
        # Metrics are still logged to MLflow via log_metrics() in the train loop
        logger.debug(f"Skipping MLflow model artifact logging for epoch {epoch}")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', {})

        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint