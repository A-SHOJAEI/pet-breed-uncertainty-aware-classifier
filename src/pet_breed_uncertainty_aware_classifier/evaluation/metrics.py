"""Comprehensive evaluation metrics for uncertainty-aware classification."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


class CalibrationError:
    """Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) computation."""

    def __init__(self, n_bins: int = 15):
        """
        Initialize calibration error calculator.

        Args:
            n_bins: Number of bins for calibration curve
        """
        self.n_bins = n_bins

    def compute_ece(
        self,
        predictions: Union[Tensor, np.ndarray],
        targets: Union[Tensor, np.ndarray],
        confidences: Optional[Union[Tensor, np.ndarray]] = None
    ) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            predictions: Model predictions (probabilities or logits)
            targets: Ground truth labels
            confidences: Confidence scores (if None, uses max probability)

        Returns:
            ECE value
        """
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, Tensor):
            targets = targets.cpu().numpy()

        # Convert logits to probabilities if needed
        if predictions.max() > 1.0:
            predictions = F.softmax(torch.from_numpy(predictions), dim=1).numpy()

        # Use max probability as confidence if not provided
        if confidences is None:
            confidences = np.max(predictions, axis=1)
        elif isinstance(confidences, Tensor):
            confidences = confidences.cpu().numpy()

        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        accuracies = (predicted_classes == targets).astype(float)

        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def compute_mce(
        self,
        predictions: Union[Tensor, np.ndarray],
        targets: Union[Tensor, np.ndarray],
        confidences: Optional[Union[Tensor, np.ndarray]] = None
    ) -> float:
        """
        Compute Maximum Calibration Error.

        Args:
            predictions: Model predictions (probabilities or logits)
            targets: Ground truth labels
            confidences: Confidence scores (if None, uses max probability)

        Returns:
            MCE value
        """
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, Tensor):
            targets = targets.cpu().numpy()

        # Convert logits to probabilities if needed
        if predictions.max() > 1.0:
            predictions = F.softmax(torch.from_numpy(predictions), dim=1).numpy()

        # Use max probability as confidence if not provided
        if confidences is None:
            confidences = np.max(predictions, axis=1)
        elif isinstance(confidences, Tensor):
            confidences = confidences.cpu().numpy()

        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        accuracies = (predicted_classes == targets).astype(float)

        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = max(mce, calibration_error)

        return mce

    def reliability_diagram_data(
        self,
        predictions: Union[Tensor, np.ndarray],
        targets: Union[Tensor, np.ndarray],
        confidences: Optional[Union[Tensor, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate data for reliability diagram plotting.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            confidences: Confidence scores

        Returns:
            Dictionary with bin data for plotting
        """
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, Tensor):
            targets = targets.cpu().numpy()

        # Convert logits to probabilities if needed
        if predictions.max() > 1.0:
            predictions = F.softmax(torch.from_numpy(predictions), dim=1).numpy()

        # Use max probability as confidence if not provided
        if confidences is None:
            confidences = np.max(predictions, axis=1)
        elif isinstance(confidences, Tensor):
            confidences = confidences.cpu().numpy()

        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        accuracies = (predicted_classes == targets).astype(float)

        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_count = in_bin.sum()

            if bin_count > 0:
                bin_confidence = confidences[in_bin].mean()
                bin_accuracy = accuracies[in_bin].mean()
            else:
                bin_confidence = (bin_lower + bin_upper) / 2
                bin_accuracy = 0.0

            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(bin_count)

        return {
            'confidences': np.array(bin_confidences),
            'accuracies': np.array(bin_accuracies),
            'counts': np.array(bin_counts),
            'bin_boundaries': bin_boundaries
        }


class ReliabilityDiagram:
    """Reliability diagram visualization for calibration assessment."""

    def __init__(self, calibration_error: CalibrationError):
        """
        Initialize reliability diagram.

        Args:
            calibration_error: CalibrationError instance
        """
        self.calibration_error = calibration_error

    def plot(
        self,
        predictions: Union[Tensor, np.ndarray],
        targets: Union[Tensor, np.ndarray],
        confidences: Optional[Union[Tensor, np.ndarray]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot reliability diagram.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            confidences: Confidence scores
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        # Get bin data
        bin_data = self.calibration_error.reliability_diagram_data(
            predictions, targets, confidences
        )

        # Calculate ECE and MCE
        ece = self.calibration_error.compute_ece(predictions, targets, confidences)
        mce = self.calibration_error.compute_mce(predictions, targets, confidences)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot bars
        bar_width = 1.0 / self.calibration_error.n_bins
        x_positions = bin_data['confidences']

        bars = ax.bar(
            x_positions,
            bin_data['accuracies'],
            width=bar_width * 0.8,
            alpha=0.7,
            color='skyblue',
            edgecolor='navy',
            label='Accuracy'
        )

        # Color bars based on sample count
        max_count = bin_data['counts'].max()
        for bar, count in zip(bars, bin_data['counts']):
            if max_count > 0:
                intensity = count / max_count
                bar.set_alpha(0.3 + 0.7 * intensity)

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')

        # Plot bin confidence points
        ax.plot(
            bin_data['confidences'],
            bin_data['accuracies'],
            'ko-',
            markersize=6,
            linewidth=2,
            label='Model Calibration'
        )

        # Formatting
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Reliability Diagram\nECE: {ece:.3f}, MCE: {mce:.3f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class OODDetectionMetrics:
    """Out-of-distribution detection metrics."""

    @staticmethod
    def compute_auroc(
        in_distribution_scores: Union[Tensor, np.ndarray],
        ood_scores: Union[Tensor, np.ndarray],
        score_type: str = "confidence"
    ) -> float:
        """
        Compute AUROC for OOD detection.

        Args:
            in_distribution_scores: Scores for in-distribution samples
            ood_scores: Scores for OOD samples
            score_type: Type of score ("confidence" or "uncertainty")

        Returns:
            AUROC value
        """
        if isinstance(in_distribution_scores, Tensor):
            in_distribution_scores = in_distribution_scores.cpu().numpy()
        if isinstance(ood_scores, Tensor):
            ood_scores = ood_scores.cpu().numpy()

        # Create labels (1 for in-distribution, 0 for OOD)
        in_dist_labels = np.ones(len(in_distribution_scores))
        ood_labels = np.zeros(len(ood_scores))

        all_scores = np.concatenate([in_distribution_scores, ood_scores])
        all_labels = np.concatenate([in_dist_labels, ood_labels])

        # For uncertainty scores, flip the labels since higher uncertainty
        # should indicate OOD
        if score_type == "uncertainty":
            all_labels = 1 - all_labels

        return roc_auc_score(all_labels, all_scores)

    @staticmethod
    def compute_aupr(
        in_distribution_scores: Union[Tensor, np.ndarray],
        ood_scores: Union[Tensor, np.ndarray],
        score_type: str = "confidence"
    ) -> float:
        """
        Compute AUPR for OOD detection.

        Args:
            in_distribution_scores: Scores for in-distribution samples
            ood_scores: Scores for OOD samples
            score_type: Type of score ("confidence" or "uncertainty")

        Returns:
            AUPR value
        """
        if isinstance(in_distribution_scores, Tensor):
            in_distribution_scores = in_distribution_scores.cpu().numpy()
        if isinstance(ood_scores, Tensor):
            ood_scores = ood_scores.cpu().numpy()

        # Create labels
        in_dist_labels = np.ones(len(in_distribution_scores))
        ood_labels = np.zeros(len(ood_scores))

        all_scores = np.concatenate([in_distribution_scores, ood_scores])
        all_labels = np.concatenate([in_dist_labels, ood_labels])

        if score_type == "uncertainty":
            all_labels = 1 - all_labels

        return average_precision_score(all_labels, all_scores)


class UncertaintyMetrics:
    """Comprehensive uncertainty quantification metrics."""

    def __init__(self):
        """Initialize uncertainty metrics calculator."""
        self.calibration_error = CalibrationError()
        self.ood_metrics = OODDetectionMetrics()

    def compute_all_metrics(
        self,
        predictions: Union[Tensor, np.ndarray],
        targets: Union[Tensor, np.ndarray],
        uncertainties: Optional[Union[Tensor, np.ndarray]] = None,
        confidences: Optional[Union[Tensor, np.ndarray]] = None,
        ood_predictions: Optional[Union[Tensor, np.ndarray]] = None,
        ood_uncertainties: Optional[Union[Tensor, np.ndarray]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            predictions: Model predictions (probabilities or logits)
            targets: Ground truth labels
            uncertainties: Uncertainty scores
            confidences: Confidence scores
            ood_predictions: Out-of-distribution predictions
            ood_uncertainties: OOD uncertainty scores
            class_names: List of class names

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Convert tensors to numpy
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, Tensor):
            targets = targets.cpu().numpy()
        if isinstance(uncertainties, Tensor):
            uncertainties = uncertainties.cpu().numpy()
        if isinstance(confidences, Tensor):
            confidences = confidences.cpu().numpy()
        if isinstance(ood_predictions, Tensor):
            ood_predictions = ood_predictions.cpu().numpy()
        if isinstance(ood_uncertainties, Tensor):
            ood_uncertainties = ood_uncertainties.cpu().numpy()

        # Convert logits to probabilities if needed
        if predictions.max() > 1.0:
            predictions = F.softmax(torch.from_numpy(predictions), dim=1).numpy()

        # Basic classification metrics
        predicted_classes = np.argmax(predictions, axis=1)

        # Accuracy metrics
        metrics['accuracy'] = accuracy_score(targets, predicted_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predicted_classes, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = \
            precision_recall_fscore_support(targets, predicted_classes, average=None, zero_division=0)

        metrics['per_class'] = {
            'precision': per_class_precision,
            'recall': per_class_recall,
            'f1_score': per_class_f1,
            'support': support
        }

        if class_names:
            metrics['per_class']['class_names'] = class_names

        # Top-k accuracy
        metrics['top5_accuracy'] = self._compute_topk_accuracy(predictions, targets, k=5)

        # Confidence-based metrics
        if confidences is None:
            confidences = np.max(predictions, axis=1)
        elif isinstance(confidences, Tensor):
            confidences = confidences.cpu().numpy()

        # Calibration metrics
        metrics['expected_calibration_error'] = self.calibration_error.compute_ece(
            predictions, targets, confidences
        )
        metrics['maximum_calibration_error'] = self.calibration_error.compute_mce(
            predictions, targets, confidences
        )

        # Brier score
        targets_one_hot = np.eye(predictions.shape[1])[targets]
        metrics['brier_score'] = brier_score_loss(
            targets_one_hot.ravel(),
            predictions.ravel()
        )

        # Uncertainty correlation metrics
        if uncertainties is not None:
            if isinstance(uncertainties, Tensor):
                uncertainties = uncertainties.cpu().numpy()

            # Correlation between uncertainty and correctness
            correct_predictions = (predicted_classes == targets).astype(float)
            uncertainty_error_correlation = np.corrcoef(uncertainties, 1 - correct_predictions)[0, 1]
            metrics['uncertainty_error_correlation'] = uncertainty_error_correlation

            # Area under the rejection curve
            metrics['auroc_rejection'] = self._compute_rejection_auroc(
                correct_predictions, uncertainties
            )

        # OOD detection metrics
        if ood_predictions is not None:
            in_dist_confidences = confidences
            ood_confidences = np.max(ood_predictions, axis=1) if ood_predictions.ndim > 1 else ood_predictions

            metrics['ood_auroc_confidence'] = self.ood_metrics.compute_auroc(
                in_dist_confidences, ood_confidences, score_type="confidence"
            )
            metrics['ood_aupr_confidence'] = self.ood_metrics.compute_aupr(
                in_dist_confidences, ood_confidences, score_type="confidence"
            )

            if ood_uncertainties is not None and uncertainties is not None:
                metrics['ood_auroc_uncertainty'] = self.ood_metrics.compute_auroc(
                    uncertainties, ood_uncertainties, score_type="uncertainty"
                )
                metrics['ood_aupr_uncertainty'] = self.ood_metrics.compute_aupr(
                    uncertainties, ood_uncertainties, score_type="uncertainty"
                )

        return metrics

    def _compute_topk_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        k: int = 5
    ) -> float:
        """Compute top-k accuracy."""
        if predictions.shape[1] < k:
            k = predictions.shape[1]

        top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
        correct = np.any(top_k_predictions == targets.reshape(-1, 1), axis=1)
        return correct.mean()

    def _compute_rejection_auroc(
        self,
        correct_predictions: np.ndarray,
        uncertainties: np.ndarray
    ) -> float:
        """Compute AUROC for selective prediction (rejection)."""
        # Higher uncertainty should correspond to incorrect predictions
        return roc_auc_score(1 - correct_predictions, uncertainties)

    def print_summary(self, metrics: Dict[str, Union[float, np.ndarray, Dict]]) -> None:
        """
        Print a formatted summary of metrics.

        Args:
            metrics: Dictionary of computed metrics
        """
        print("\n" + "="*60)
        print("UNCERTAINTY-AWARE CLASSIFICATION METRICS")
        print("="*60)

        # Basic classification metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Top-5 Accuracy: {metrics.get('top5_accuracy', 'N/A'):.4f}")
        print(f"Precision (weighted): {metrics['precision']:.4f}")
        print(f"Recall (weighted): {metrics['recall']:.4f}")
        print(f"F1 Score (weighted): {metrics['f1_score']:.4f}")

        # Calibration metrics
        print(f"\nCalibration:")
        print(f"  Expected Calibration Error: {metrics['expected_calibration_error']:.4f}")
        print(f"  Maximum Calibration Error: {metrics['maximum_calibration_error']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")

        # Uncertainty metrics
        if 'uncertainty_error_correlation' in metrics:
            print(f"\nUncertainty Quality:")
            print(f"  Uncertainty-Error Correlation: {metrics['uncertainty_error_correlation']:.4f}")
            print(f"  AUROC (Rejection): {metrics['auroc_rejection']:.4f}")

        # OOD detection metrics
        if 'ood_auroc_confidence' in metrics:
            print(f"\nOOD Detection:")
            print(f"  AUROC (Confidence): {metrics['ood_auroc_confidence']:.4f}")
            print(f"  AUPR (Confidence): {metrics['ood_aupr_confidence']:.4f}")

            if 'ood_auroc_uncertainty' in metrics:
                print(f"  AUROC (Uncertainty): {metrics['ood_auroc_uncertainty']:.4f}")
                print(f"  AUPR (Uncertainty): {metrics['ood_aupr_uncertainty']:.4f}")

        print("="*60)