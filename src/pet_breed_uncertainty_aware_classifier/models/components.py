"""Custom neural network components for uncertainty-aware classification.

This module implements specialized layers and blocks that enhance uncertainty
quantification in deep learning models.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (Drop Path) for regularization and uncertainty estimation.

    During training, randomly drops entire residual branches with probability `drop_prob`.
    This provides an implicit ensemble effect and improves uncertainty calibration.

    Reference: Deep Networks with Stochastic Depth (Huang et al., 2016)
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """
        Initialize Stochastic Depth layer.

        Args:
            drop_prob: Probability of dropping the residual path
            scale_by_keep: Whether to scale by keep probability during training
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth."""
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        if self.scale_by_keep and keep_prob > 0.0:
            x = x.div(keep_prob)

        return x * random_tensor

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.3f}'


class SpectralNormalization(nn.Module):
    """
    Spectral Normalization wrapper for improving model calibration.

    Constrains the Lipschitz constant of the network, which helps prevent
    overconfident predictions and improves uncertainty estimates.

    Reference: Spectral Normalization for Generative Adversarial Networks (Miyato et al., 2018)
    """

    def __init__(
        self,
        module: nn.Module,
        name: str = 'weight',
        n_power_iterations: int = 1,
        eps: float = 1e-12
    ):
        """
        Initialize Spectral Normalization.

        Args:
            module: Module to apply spectral normalization to
            name: Name of weight parameter
            n_power_iterations: Number of power iterations
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        # Register spectral norm parameters
        if not hasattr(module, name):
            raise ValueError(f"Module {module} has no parameter {name}")

        weight = getattr(module, name)
        with torch.no_grad():
            height = weight.shape[0]
            width = weight.view(height, -1).shape[1]

            u = weight.new_empty(height).normal_(0, 1)
            v = weight.new_empty(width).normal_(0, 1)
            u = F.normalize(u, dim=0, eps=self.eps)
            v = F.normalize(v, dim=0, eps=self.eps)

        self.register_buffer(name + "_u", u)
        self.register_buffer(name + "_v", v)
        self.register_buffer(name + "_orig", weight.detach().clone())

    def forward(self, *args, **kwargs):
        """Apply spectral normalization and forward pass."""
        self._update_u_v()
        return self.module(*args, **kwargs)

    def _update_u_v(self):
        """Update u and v vectors using power iteration."""
        u = getattr(self, self.name + "_u")
        v = getattr(self, self.name + "_v")
        weight = getattr(self.module, self.name)

        height = weight.shape[0]
        weight_mat = weight.view(height, -1)

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight_normalized = weight / sigma
        setattr(self.module, self.name, weight_normalized)


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for model calibration.

    Learns a single temperature parameter to scale logits, improving the
    calibration of probability predictions without changing accuracy.

    Reference: On Calibration of Modern Neural Networks (Guo et al., 2017)
    """

    def __init__(self, init_temperature: float = 1.5):
        """
        Initialize temperature scaling.

        Args:
            init_temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: Tensor) -> Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Model logits (before softmax)

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


class EvidentialLayer(nn.Module):
    """
    Evidential Deep Learning output layer for uncertainty quantification.

    Instead of predicting class probabilities directly, predicts Dirichlet
    distribution parameters (evidence), enabling both aleatoric and epistemic
    uncertainty estimation.

    Reference: Evidential Deep Learning to Quantify Classification Uncertainty (Sensoy et al., 2018)
    """

    def __init__(self, in_features: int, num_classes: int):
        """
        Initialize evidential layer.

        Args:
            in_features: Number of input features
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> dict:
        """
        Forward pass producing evidential outputs.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing:
            - evidence: Evidence for each class
            - alpha: Dirichlet parameters
            - S: Total evidence
            - prob: Expected probabilities
            - uncertainty: Uncertainty estimate
        """
        logits = self.linear(x)
        evidence = F.softplus(logits)  # Ensure positive evidence
        alpha = evidence + 1  # Dirichlet parameters

        S = torch.sum(alpha, dim=1, keepdim=True)  # Total evidence
        prob = alpha / S  # Expected probability

        # Uncertainty based on lack of evidence
        uncertainty = self.num_classes / S

        return {
            'evidence': evidence,
            'alpha': alpha,
            'S': S,
            'prob': prob,
            'uncertainty': uncertainty
        }


class AdaptiveDropout(nn.Module):
    """
    Adaptive Dropout that adjusts dropout rate based on uncertainty.

    Higher dropout is applied when the model is uncertain, encouraging
    more stochastic behavior for difficult examples.
    """

    def __init__(
        self,
        base_dropout: float = 0.3,
        min_dropout: float = 0.1,
        max_dropout: float = 0.5
    ):
        """
        Initialize adaptive dropout.

        Args:
            base_dropout: Base dropout probability
            min_dropout: Minimum dropout rate
            max_dropout: Maximum dropout rate
        """
        super().__init__()
        self.base_dropout = base_dropout
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout

    def forward(self, x: Tensor, uncertainty: Optional[Tensor] = None) -> Tensor:
        """
        Apply adaptive dropout.

        Args:
            x: Input tensor
            uncertainty: Optional uncertainty estimates (batch_size,)

        Returns:
            Dropout-regularized tensor
        """
        if not self.training:
            return x

        if uncertainty is None:
            # Use base dropout if no uncertainty provided
            return F.dropout(x, p=self.base_dropout, training=True)

        # Scale dropout based on uncertainty
        uncertainty_scaled = uncertainty.view(-1, 1)  # Ensure proper shape
        dropout_rate = self.min_dropout + (self.max_dropout - self.min_dropout) * uncertainty_scaled
        dropout_rate = torch.clamp(dropout_rate, self.min_dropout, self.max_dropout)

        # Apply element-wise dropout with varying rates
        mask = torch.bernoulli(1 - dropout_rate.expand_as(x))
        return x * mask / (1 - dropout_rate.expand_as(x))


class FeatureCalibration(nn.Module):
    """
    Feature-level calibration layer.

    Applies learnable scaling and bias to intermediate features to improve
    downstream calibration without changing feature extractors.
    """

    def __init__(self, num_features: int):
        """
        Initialize feature calibration.

        Args:
            num_features: Number of input features
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feature calibration.

        Args:
            x: Input features

        Returns:
            Calibrated features
        """
        return x * self.scale + self.bias


class UncertaintyGuidedAttention(nn.Module):
    """
    Attention mechanism that incorporates uncertainty estimates.

    Attends more to confident predictions and suppresses uncertain ones,
    improving robustness in ensemble or multi-view scenarios.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4):
        """
        Initialize uncertainty-guided attention.

        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor, uncertainty: Optional[Tensor] = None) -> Tensor:
        """
        Apply uncertainty-guided attention.

        Args:
            x: Input features (batch, seq_len, feature_dim)
            uncertainty: Uncertainty scores (batch, seq_len)

        Returns:
            Attended features
        """
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Modulate by uncertainty if provided
        if uncertainty is not None:
            # High uncertainty -> lower attention weights
            uncertainty_weight = 1.0 / (1.0 + uncertainty.unsqueeze(1).unsqueeze(-1))
            attn = attn * uncertainty_weight

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)

        return x
