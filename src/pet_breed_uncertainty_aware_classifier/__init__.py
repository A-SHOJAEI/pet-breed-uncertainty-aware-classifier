"""
Pet Breed Uncertainty-Aware Classifier

A fine-grained pet breed classifier that implements Monte Carlo Dropout and deep ensembles
to quantify prediction uncertainty for real-world deployment scenarios.
"""

__version__ = "0.1.0"
__author__ = "ML Engineering Team"

from .models.model import UncertaintyAwareClassifier
from .data.loader import PetDataLoader
from .training.trainer import UncertaintyTrainer
from .evaluation.metrics import CalibrationError, UncertaintyMetrics

__all__ = [
    "UncertaintyAwareClassifier",
    "PetDataLoader",
    "UncertaintyTrainer",
    "CalibrationError",
    "UncertaintyMetrics",
]