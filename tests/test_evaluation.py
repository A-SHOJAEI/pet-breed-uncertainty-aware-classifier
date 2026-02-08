"""Tests for evaluation metrics."""

import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import sys

from pet_breed_uncertainty_aware_classifier.evaluation.metrics import (
    CalibrationError, UncertaintyMetrics, ReliabilityDiagram, OODDetectionMetrics
)


class TestCalibrationError:
    """Test calibration error computation."""

    def test_calibration_error_initialization(self):
        """Test CalibrationError initialization."""
        cal_error = CalibrationError(n_bins=10)
        assert cal_error.n_bins == 10

        cal_error_default = CalibrationError()
        assert cal_error_default.n_bins == 15

    def test_ece_computation_perfect_calibration(self):
        """Test ECE computation with perfect calibration."""
        cal_error = CalibrationError(n_bins=10)

        # Perfect calibration: confidence equals accuracy
        predictions = torch.zeros(100, 5)
        targets = torch.zeros(100, dtype=torch.long)

        # Set predictions so max probability equals accuracy
        for i in range(100):
            conf = i / 100.0  # Confidence from 0 to 1
            predictions[i, 0] = conf
            predictions[i, 1] = 1 - conf

            # Set target based on confidence (perfect calibration)
            if conf > 0.5:
                targets[i] = 0
            else:
                targets[i] = 1

        # Convert to probabilities
        predictions = torch.softmax(predictions * 5, dim=1)  # Scale to make differences more pronounced

        ece = cal_error.compute_ece(predictions, targets)

        # ECE should be low for well-calibrated model
        assert isinstance(ece, float)
        assert ece >= 0

    def test_ece_computation_with_confidences(self):
        """Test ECE computation with provided confidences."""
        cal_error = CalibrationError(n_bins=5)

        predictions = torch.rand(50, 10)
        predictions = torch.softmax(predictions, dim=1)
        targets = torch.randint(0, 10, (50,))
        confidences = torch.rand(50)

        ece = cal_error.compute_ece(predictions, targets, confidences)

        assert isinstance(ece, float)
        assert ece >= 0
        assert ece <= 1  # ECE should be between 0 and 1

    def test_mce_computation(self):
        """Test MCE computation."""
        cal_error = CalibrationError(n_bins=10)

        predictions = torch.rand(100, 5)
        predictions = torch.softmax(predictions, dim=1)
        targets = torch.randint(0, 5, (100,))

        mce = cal_error.compute_mce(predictions, targets)

        assert isinstance(mce, float)
        assert mce >= 0
        assert mce <= 1

    def test_reliability_diagram_data(self):
        """Test reliability diagram data generation."""
        cal_error = CalibrationError(n_bins=5)

        predictions = torch.rand(100, 3)
        predictions = torch.softmax(predictions, dim=1)
        targets = torch.randint(0, 3, (100,))

        bin_data = cal_error.reliability_diagram_data(predictions, targets)

        assert 'confidences' in bin_data
        assert 'accuracies' in bin_data
        assert 'counts' in bin_data
        assert 'bin_boundaries' in bin_data

        assert len(bin_data['confidences']) == cal_error.n_bins
        assert len(bin_data['accuracies']) == cal_error.n_bins
        assert len(bin_data['counts']) == cal_error.n_bins
        assert len(bin_data['bin_boundaries']) == cal_error.n_bins + 1

    def test_with_numpy_arrays(self):
        """Test calibration error with numpy arrays."""
        cal_error = CalibrationError()

        predictions = np.random.rand(50, 5)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)  # Normalize
        targets = np.random.randint(0, 5, 50)

        ece = cal_error.compute_ece(predictions, targets)
        mce = cal_error.compute_mce(predictions, targets)

        assert isinstance(ece, float)
        assert isinstance(mce, float)
        assert ece >= 0 and mce >= 0

    def test_with_logits(self):
        """Test calibration error computation with logits."""
        cal_error = CalibrationError()

        # Create logits (values > 1)
        logits = torch.randn(50, 5) * 5  # Scale to ensure some values > 1
        targets = torch.randint(0, 5, (50,))

        ece = cal_error.compute_ece(logits, targets)

        assert isinstance(ece, float)
        assert ece >= 0


class TestReliabilityDiagram:
    """Test reliability diagram plotting."""

    def test_reliability_diagram_initialization(self):
        """Test reliability diagram initialization."""
        cal_error = CalibrationError()
        rel_diagram = ReliabilityDiagram(cal_error)

        assert rel_diagram.calibration_error == cal_error

    def test_reliability_diagram_plot(self):
        """Test reliability diagram plotting."""
        cal_error = CalibrationError(n_bins=5)
        rel_diagram = ReliabilityDiagram(cal_error)

        predictions = torch.rand(100, 5)
        predictions = torch.softmax(predictions, dim=1)
        targets = torch.randint(0, 5, (100,))

        # Test plotting without saving
        fig = rel_diagram.plot(predictions, targets)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up

    def test_reliability_diagram_plot_with_save(self, temp_dir):
        """Test reliability diagram plotting with saving."""
        cal_error = CalibrationError(n_bins=5)
        rel_diagram = ReliabilityDiagram(cal_error)

        predictions = torch.rand(100, 5)
        predictions = torch.softmax(predictions, dim=1)
        targets = torch.randint(0, 5, (100,))

        save_path = temp_dir / "test_reliability.png"
        fig = rel_diagram.plot(predictions, targets, save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)


class TestOODDetectionMetrics:
    """Test out-of-distribution detection metrics."""

    def test_auroc_confidence_scores(self):
        """Test AUROC computation with confidence scores."""
        in_dist_scores = torch.rand(100) * 0.5 + 0.5  # High confidence [0.5, 1.0]
        ood_scores = torch.rand(100) * 0.5  # Low confidence [0.0, 0.5]

        auroc = OODDetectionMetrics.compute_auroc(
            in_dist_scores, ood_scores, score_type="confidence"
        )

        assert isinstance(auroc, float)
        assert 0 <= auroc <= 1
        assert auroc > 0.5  # Should be better than random for this setup

    def test_auroc_uncertainty_scores(self):
        """Test AUROC computation with uncertainty scores."""
        in_dist_scores = torch.rand(100) * 0.5  # Low uncertainty
        ood_scores = torch.rand(100) * 0.5 + 0.5  # High uncertainty

        auroc = OODDetectionMetrics.compute_auroc(
            in_dist_scores, ood_scores, score_type="uncertainty"
        )

        assert isinstance(auroc, float)
        assert 0 <= auroc <= 1

    def test_aupr_computation(self):
        """Test AUPR computation."""
        in_dist_scores = torch.rand(50) + 0.5  # Higher scores
        ood_scores = torch.rand(50)  # Lower scores

        aupr = OODDetectionMetrics.compute_aupr(
            in_dist_scores, ood_scores, score_type="confidence"
        )

        assert isinstance(aupr, float)
        assert 0 <= aupr <= 1

    def test_with_numpy_arrays(self):
        """Test OOD metrics with numpy arrays."""
        in_dist_scores = np.random.rand(50) + 0.5
        ood_scores = np.random.rand(50)

        auroc = OODDetectionMetrics.compute_auroc(
            in_dist_scores, ood_scores, score_type="confidence"
        )
        aupr = OODDetectionMetrics.compute_aupr(
            in_dist_scores, ood_scores, score_type="confidence"
        )

        assert isinstance(auroc, float)
        assert isinstance(aupr, float)


class TestUncertaintyMetrics:
    """Test comprehensive uncertainty metrics."""

    def test_uncertainty_metrics_initialization(self):
        """Test UncertaintyMetrics initialization."""
        metrics = UncertaintyMetrics()

        assert isinstance(metrics.calibration_error, CalibrationError)
        assert isinstance(metrics.ood_metrics, OODDetectionMetrics)

    def test_compute_all_metrics_basic(self, sample_predictions):
        """Test basic metrics computation."""
        metrics = UncertaintyMetrics()

        results = metrics.compute_all_metrics(
            predictions=sample_predictions['predictions'],
            targets=sample_predictions['targets'],
            uncertainties=sample_predictions['uncertainties'],
            confidences=sample_predictions['confidences']
        )

        # Check basic classification metrics
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        assert all(metric in results for metric in basic_metrics)

        # Check calibration metrics
        calibration_metrics = ['expected_calibration_error', 'maximum_calibration_error', 'brier_score']
        assert all(metric in results for metric in calibration_metrics)

        # Check uncertainty metrics
        uncertainty_metrics = ['uncertainty_error_correlation', 'auroc_rejection']
        assert all(metric in results for metric in uncertainty_metrics)

        # Check per-class metrics
        assert 'per_class' in results
        assert 'precision' in results['per_class']
        assert 'recall' in results['per_class']
        assert 'f1_score' in results['per_class']
        assert 'support' in results['per_class']

    def test_compute_all_metrics_with_ood(self, sample_predictions):
        """Test metrics computation with OOD data."""
        metrics = UncertaintyMetrics()

        # Create OOD data
        ood_predictions = torch.rand(50, 5)
        ood_predictions = torch.softmax(ood_predictions, dim=1)
        ood_uncertainties = torch.rand(50) + 0.5  # Higher uncertainty for OOD

        results = metrics.compute_all_metrics(
            predictions=sample_predictions['predictions'],
            targets=sample_predictions['targets'],
            uncertainties=sample_predictions['uncertainties'],
            confidences=sample_predictions['confidences'],
            ood_predictions=ood_predictions,
            ood_uncertainties=ood_uncertainties
        )

        # Check OOD metrics are computed
        ood_metrics = [
            'ood_auroc_confidence', 'ood_aupr_confidence',
            'ood_auroc_uncertainty', 'ood_aupr_uncertainty'
        ]
        assert all(metric in results for metric in ood_metrics)

    def test_compute_all_metrics_with_class_names(self, sample_predictions):
        """Test metrics computation with class names."""
        metrics = UncertaintyMetrics()
        class_names = [f"Class_{i}" for i in range(5)]

        results = metrics.compute_all_metrics(
            predictions=sample_predictions['predictions'],
            targets=sample_predictions['targets'],
            class_names=class_names
        )

        assert 'per_class' in results
        assert 'class_names' in results['per_class']
        assert results['per_class']['class_names'] == class_names

    def test_topk_accuracy(self, sample_predictions):
        """Test top-k accuracy computation."""
        metrics = UncertaintyMetrics()

        # Test top-5 accuracy
        top5_acc = metrics._compute_topk_accuracy(
            sample_predictions['predictions'].numpy(),
            sample_predictions['targets'].numpy(),
            k=5
        )

        assert isinstance(top5_acc, float)
        assert 0 <= top5_acc <= 1

        # Top-5 accuracy should be >= top-1 accuracy
        top1_acc = metrics._compute_topk_accuracy(
            sample_predictions['predictions'].numpy(),
            sample_predictions['targets'].numpy(),
            k=1
        )

        assert top5_acc >= top1_acc

    def test_rejection_auroc(self):
        """Test rejection AUROC computation."""
        metrics = UncertaintyMetrics()

        # Create synthetic data where higher uncertainty correlates with errors
        correct_predictions = np.array([1, 1, 1, 0, 0, 0])  # First 3 correct, last 3 incorrect
        uncertainties = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])  # Lower uncertainty for correct

        auroc = metrics._compute_rejection_auroc(correct_predictions, uncertainties)

        assert isinstance(auroc, float)
        assert 0 <= auroc <= 1

    def test_print_summary(self, sample_predictions):
        """Test metrics summary printing."""
        metrics = UncertaintyMetrics()

        results = metrics.compute_all_metrics(
            predictions=sample_predictions['predictions'],
            targets=sample_predictions['targets'],
            uncertainties=sample_predictions['uncertainties'],
            confidences=sample_predictions['confidences']
        )

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        metrics.print_summary(results)

        # Reset stdout
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # Check that key metrics are printed
        assert "UNCERTAINTY-AWARE CLASSIFICATION METRICS" in output
        assert "Accuracy:" in output
        assert "Expected Calibration Error:" in output
        assert "Uncertainty-Error Correlation:" in output

    def test_metrics_with_edge_cases(self):
        """Test metrics computation with edge cases."""
        metrics = UncertaintyMetrics()

        # Test with perfect predictions
        targets = torch.repeat_interleave(torch.arange(5), 10)
        predictions = torch.zeros(50, 5)
        predictions[torch.arange(50), targets] = 1.0  # Perfect one-hot predictions
        uncertainties = torch.zeros(50)  # Zero uncertainty
        confidences = torch.ones(50)  # Perfect confidence

        results = metrics.compute_all_metrics(predictions, targets, uncertainties, confidences)

        # Accuracy should be perfect
        assert results['accuracy'] == 1.0

        # ECE should be low for perfect calibration
        assert results['expected_calibration_error'] <= 0.1

    def test_brier_score_computation(self, sample_predictions):
        """Test Brier score computation."""
        metrics = UncertaintyMetrics()

        results = metrics.compute_all_metrics(
            predictions=sample_predictions['predictions'],
            targets=sample_predictions['targets']
        )

        assert 'brier_score' in results
        assert isinstance(results['brier_score'], float)
        assert 0 <= results['brier_score'] <= 1

    def test_metrics_value_ranges(self, sample_predictions):
        """Test that all metrics are in expected ranges."""
        metrics = UncertaintyMetrics()

        results = metrics.compute_all_metrics(
            predictions=sample_predictions['predictions'],
            targets=sample_predictions['targets'],
            uncertainties=sample_predictions['uncertainties'],
            confidences=sample_predictions['confidences']
        )

        # Check value ranges
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1
        assert 0 <= results['expected_calibration_error'] <= 1
        assert 0 <= results['maximum_calibration_error'] <= 1
        assert 0 <= results['brier_score'] <= 1

        # Correlation can be negative
        assert -1 <= results['uncertainty_error_correlation'] <= 1

        # AUROC should be between 0 and 1
        assert 0 <= results['auroc_rejection'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])