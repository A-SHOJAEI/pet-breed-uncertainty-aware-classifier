"""Uncertainty-aware models for pet breed classification."""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import timm
import numpy as np

from ..utils.config import ModelConfig


logger = logging.getLogger(__name__)


class MCDropout(nn.Module):
    """Monte Carlo Dropout layer that stays active during inference."""

    def __init__(self, p: float = 0.5):
        """
        Initialize MC Dropout layer.

        Args:
            p: Dropout probability
        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout regardless of training mode."""
        return F.dropout(x, p=self.p, training=True)


class UncertaintyHead(nn.Module):
    """Classification head with uncertainty quantification capabilities."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout_rate: float = 0.3,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize uncertainty head.

        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            dropout_rate: Dropout rate for MC Dropout
            hidden_dim: Hidden layer dimension (if None, uses in_features // 2)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(in_features // 2, num_classes * 2)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            MCDropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            MCDropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize layer weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through classification head."""
        return self.classifier(x)


class MCDropoutClassifier(nn.Module):
    """Base classifier with Monte Carlo Dropout for uncertainty estimation."""

    def __init__(self, config: ModelConfig):
        """
        Initialize MC Dropout classifier.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Load pre-trained backbone
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )

        # Freeze backbone if requested
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Create uncertainty-aware classification head
        self.head = UncertaintyHead(
            in_features=self.feature_dim,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def predict_with_uncertainty(
        self,
        x: Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout.

        Args:
            x: Input tensor
            num_samples: Number of MC samples (if None, uses config value)

        Returns:
            Tuple of (mean_predictions, uncertainty, all_predictions)
        """
        if num_samples is None:
            num_samples = self.config.mc_samples

        # Enable train mode for MC Dropout
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(F.softmax(pred, dim=1))

        # Stack predictions
        all_predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, num_classes)

        # Calculate mean and uncertainty
        mean_pred = torch.mean(all_predictions, dim=0)
        uncertainty = torch.var(all_predictions, dim=0)

        # Return to original mode
        self.eval()

        return mean_pred, uncertainty, all_predictions


class DeepEnsemble(nn.Module):
    """Deep ensemble of multiple models for uncertainty estimation."""

    def __init__(self, config: ModelConfig):
        """
        Initialize deep ensemble.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Create ensemble of models
        self.models = nn.ModuleList([
            MCDropoutClassifier(config)
            for _ in range(config.ensemble_size)
        ])

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass through all ensemble models.

        Args:
            x: Input tensor

        Returns:
            List of predictions from each model
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        return predictions

    def predict_with_uncertainty(
        self,
        x: Tensor,
        use_mc_dropout: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict with uncertainty using ensemble and optionally MC Dropout.

        Args:
            x: Input tensor
            use_mc_dropout: Whether to use MC Dropout within each model

        Returns:
            Tuple of (mean_predictions, uncertainty, all_predictions)
        """
        all_predictions = []

        for model in self.models:
            if use_mc_dropout:
                _, _, mc_predictions = model.predict_with_uncertainty(x)
                # Use mean of MC samples for this ensemble member
                pred = torch.mean(mc_predictions, dim=0)
            else:
                model.eval()
                with torch.no_grad():
                    pred = F.softmax(model(x), dim=1)

            all_predictions.append(pred)

        # Stack ensemble predictions
        ensemble_predictions = torch.stack(all_predictions, dim=0)

        # Calculate ensemble statistics
        mean_pred = torch.mean(ensemble_predictions, dim=0)
        uncertainty = torch.var(ensemble_predictions, dim=0)

        return mean_pred, uncertainty, ensemble_predictions


class UncertaintyAwareClassifier(nn.Module):
    """
    Main uncertainty-aware classifier combining MC Dropout and Deep Ensembles.

    This model provides multiple uncertainty estimation methods:
    1. Monte Carlo Dropout within individual models
    2. Deep ensemble across multiple models
    3. Combined approach using both methods
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize uncertainty-aware classifier.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        if config.ensemble_size > 1:
            self.model = DeepEnsemble(config)
            self.is_ensemble = True
        else:
            self.model = MCDropoutClassifier(config)
            self.is_ensemble = False

        logger.info(f"Initialized {'ensemble' if self.is_ensemble else 'single'} model "
                   f"with backbone: {config.backbone}")

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        """Forward pass through the model."""
        return self.model(x)

    def predict_with_uncertainty(
        self,
        x: Tensor,
        uncertainty_method: str = "combined",
        num_mc_samples: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        Comprehensive uncertainty prediction.

        Args:
            x: Input tensor
            uncertainty_method: 'mc_dropout', 'ensemble', or 'combined'
            num_mc_samples: Number of MC samples

        Returns:
            Dictionary containing:
            - predictions: Mean predictions
            - aleatoric_uncertainty: Data-dependent uncertainty
            - epistemic_uncertainty: Model uncertainty
            - total_uncertainty: Combined uncertainty
            - confidence: Maximum probability
            - entropy: Predictive entropy

        Raises:
            ValueError: If uncertainty_method is invalid or input is malformed
            RuntimeError: If prediction fails
        """
        # Validate inputs
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        if x.dim() != 4 or x.size(0) == 0:
            raise ValueError("Input tensor must be 4D with batch dimension > 0")

        valid_methods = ["mc_dropout", "ensemble", "combined"]
        if uncertainty_method not in valid_methods:
            raise ValueError(f"uncertainty_method must be one of {valid_methods}")

        if num_mc_samples is not None and num_mc_samples <= 0:
            raise ValueError("num_mc_samples must be positive if specified")

        try:
            if uncertainty_method == "ensemble" and not self.is_ensemble:
                logger.warning("Ensemble method requested but single model provided. Using MC Dropout.")
                uncertainty_method = "mc_dropout"

            if uncertainty_method == "mc_dropout" or not self.is_ensemble:
                mean_pred, uncertainty, all_preds = self.model.predict_with_uncertainty(
                    x, num_mc_samples
                )
                # For MC Dropout, uncertainty represents both aleatoric and epistemic
                aleatoric_uncertainty = uncertainty
                epistemic_uncertainty = uncertainty
                total_uncertainty = uncertainty

            elif uncertainty_method == "ensemble":
                mean_pred, uncertainty, all_preds = self.model.predict_with_uncertainty(
                    x, use_mc_dropout=False
                )
                # For pure ensemble, uncertainty is primarily epistemic
                aleatoric_uncertainty = torch.zeros_like(uncertainty)
                epistemic_uncertainty = uncertainty
                total_uncertainty = uncertainty

            else:  # combined
                mean_pred, uncertainty, all_preds = self.model.predict_with_uncertainty(
                    x, use_mc_dropout=True
                )
                # Combined method captures both types
                total_uncertainty = uncertainty
                # Approximate decomposition
                epistemic_uncertainty = uncertainty * 0.6
                aleatoric_uncertainty = uncertainty * 0.4

            # Calculate additional uncertainty metrics with numerical stability
            confidence = torch.max(mean_pred, dim=1)[0]
            entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)

            return {
                'predictions': mean_pred,
                'aleatoric_uncertainty': torch.mean(aleatoric_uncertainty, dim=1),
                'epistemic_uncertainty': torch.mean(epistemic_uncertainty, dim=1),
                'total_uncertainty': torch.mean(total_uncertainty, dim=1),
                'confidence': confidence,
                'entropy': entropy,
                'raw_predictions': all_preds
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Failed to compute predictions with uncertainty: {e}")

    def get_prediction_summary(
        self,
        x: Tensor,
        class_names: List[str],
        uncertainty_method: str = "combined",
        top_k: int = 3
    ) -> List[Dict]:
        """
        Get human-readable prediction summary.

        Args:
            x: Input tensor
            class_names: List of class names
            uncertainty_method: Uncertainty estimation method
            top_k: Number of top predictions to return

        Returns:
            List of prediction summaries for each input

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If prediction fails
        """
        # Validate inputs
        if not isinstance(class_names, list) or len(class_names) == 0:
            raise ValueError("class_names must be a non-empty list")

        if len(class_names) != self.config.num_classes:
            raise ValueError(f"Number of class names ({len(class_names)}) must match "
                           f"model classes ({self.config.num_classes})")

        if top_k <= 0 or top_k > len(class_names):
            raise ValueError(f"top_k must be between 1 and {len(class_names)}")

        try:
            results = self.predict_with_uncertainty(x, uncertainty_method)
            predictions = results['predictions']
            uncertainties = results['total_uncertainty']
            confidences = results['confidence']

            summaries = []
            for i in range(x.size(0)):
                pred = predictions[i]
                uncertainty = uncertainties[i].item()
                confidence = confidences[i].item()

                # Get top-k predictions
                top_probs, top_indices = torch.topk(pred, top_k)
                top_classes = [(class_names[idx.item()], prob.item())
                              for idx, prob in zip(top_indices, top_probs)]

                # Determine confidence level with bounds checking
                if confidence > 0.8 and uncertainty < 0.1:
                    conf_level = "High"
                elif confidence > 0.6 and uncertainty < 0.2:
                    conf_level = "Medium"
                else:
                    conf_level = "Low"

                summary = {
                    'top_predictions': top_classes,
                    'confidence_score': confidence,
                    'uncertainty_score': uncertainty,
                    'confidence_level': conf_level,
                    'should_review': conf_level == "Low"
                }
                summaries.append(summary)

            return summaries

        except Exception as e:
            logger.error(f"Failed to generate prediction summary: {e}")
            raise RuntimeError(f"Prediction summary generation failed: {e}")

    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'ensemble' if self.is_ensemble else 'single',
            'backbone': self.config.backbone,
            'num_classes': self.config.num_classes,
            'ensemble_size': self.config.ensemble_size if self.is_ensemble else 1,
            'mc_samples': self.config.mc_samples,
            'dropout_rate': self.config.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'pretrained': self.config.pretrained,
            'backbone_frozen': self.config.freeze_backbone
        }