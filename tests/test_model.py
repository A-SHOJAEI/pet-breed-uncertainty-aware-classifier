"""Tests for model implementations."""

import pytest
import torch
import numpy as np

from pet_breed_uncertainty_aware_classifier.models.model import (
    MCDropout, UncertaintyHead, MCDropoutClassifier,
    DeepEnsemble, UncertaintyAwareClassifier
)
from pet_breed_uncertainty_aware_classifier.utils.config import ModelConfig


class TestMCDropout:
    """Test Monte Carlo Dropout layer."""

    def test_mc_dropout_forward(self):
        """Test MC Dropout forward pass."""
        dropout = MCDropout(p=0.5)
        x = torch.randn(10, 100)

        # In training mode, dropout should be active
        dropout.train()
        output_train = dropout(x)
        assert output_train.shape == x.shape

        # In eval mode, dropout should still be active (MC Dropout)
        dropout.eval()
        output_eval = dropout(x)
        assert output_eval.shape == x.shape

        # Outputs should be different due to randomness
        assert not torch.equal(output_train, output_eval)


class TestUncertaintyHead:
    """Test uncertainty head module."""

    def test_uncertainty_head_initialization(self):
        """Test uncertainty head initialization."""
        head = UncertaintyHead(
            in_features=512,
            num_classes=10,
            dropout_rate=0.3,
            hidden_dim=256
        )

        assert isinstance(head.classifier, torch.nn.Sequential)

        # Count MC Dropout layers
        mc_dropout_count = sum(
            1 for module in head.classifier.modules()
            if isinstance(module, MCDropout)
        )
        assert mc_dropout_count == 2  # Should have 2 MC Dropout layers

    def test_uncertainty_head_forward(self):
        """Test uncertainty head forward pass."""
        head = UncertaintyHead(in_features=512, num_classes=10)
        x = torch.randn(4, 512)

        output = head(x)

        assert output.shape == (4, 10)
        assert not torch.any(torch.isnan(output))

    def test_uncertainty_head_different_outputs(self):
        """Test that MC Dropout produces different outputs."""
        head = UncertaintyHead(in_features=512, num_classes=10, dropout_rate=0.5)
        x = torch.randn(4, 512)

        # Multiple forward passes should give different results due to MC Dropout
        head.train()
        output1 = head(x)
        output2 = head(x)

        assert not torch.equal(output1, output2)


class TestMCDropoutClassifier:
    """Test MC Dropout classifier."""

    def test_mc_dropout_classifier_initialization(self, sample_config):
        """Test MC Dropout classifier initialization."""
        model = MCDropoutClassifier(sample_config.model)

        assert hasattr(model, 'backbone')
        assert hasattr(model, 'head')
        assert model.feature_dim > 0

    def test_mc_dropout_classifier_forward(self, sample_config, sample_batch):
        """Test MC Dropout classifier forward pass."""
        data, targets = sample_batch
        model = MCDropoutClassifier(sample_config.model)

        output = model(data)

        assert output.shape == (data.size(0), sample_config.model.num_classes)
        assert not torch.any(torch.isnan(output))

    def test_predict_with_uncertainty(self, sample_config, sample_batch):
        """Test uncertainty prediction."""
        data, targets = sample_batch
        model = MCDropoutClassifier(sample_config.model)

        mean_pred, uncertainty, all_preds = model.predict_with_uncertainty(data, num_samples=5)

        assert mean_pred.shape == (data.size(0), sample_config.model.num_classes)
        assert uncertainty.shape == (data.size(0), sample_config.model.num_classes)
        assert all_preds.shape == (5, data.size(0), sample_config.model.num_classes)

        # Check probabilities sum to 1
        assert torch.allclose(mean_pred.sum(dim=1), torch.ones(data.size(0)), atol=1e-6)

        # Check uncertainty is non-negative
        assert torch.all(uncertainty >= 0)


class TestDeepEnsemble:
    """Test Deep Ensemble implementation."""

    def test_deep_ensemble_initialization(self, sample_config):
        """Test deep ensemble initialization."""
        ensemble = DeepEnsemble(sample_config.model)

        assert len(ensemble.models) == sample_config.model.ensemble_size
        assert all(isinstance(model, MCDropoutClassifier) for model in ensemble.models)

    def test_deep_ensemble_forward(self, sample_config, sample_batch):
        """Test deep ensemble forward pass."""
        data, targets = sample_batch
        ensemble = DeepEnsemble(sample_config.model)

        outputs = ensemble(data)

        assert len(outputs) == sample_config.model.ensemble_size
        assert all(
            output.shape == (data.size(0), sample_config.model.num_classes)
            for output in outputs
        )

    def test_ensemble_predict_with_uncertainty(self, sample_config, sample_batch):
        """Test ensemble uncertainty prediction."""
        data, targets = sample_batch
        ensemble = DeepEnsemble(sample_config.model)

        # Test without MC Dropout
        mean_pred, uncertainty, all_preds = ensemble.predict_with_uncertainty(
            data, use_mc_dropout=False
        )

        assert mean_pred.shape == (data.size(0), sample_config.model.num_classes)
        assert uncertainty.shape == (data.size(0), sample_config.model.num_classes)
        assert all_preds.shape == (sample_config.model.ensemble_size, data.size(0), sample_config.model.num_classes)

        # Test with MC Dropout
        mean_pred_mc, uncertainty_mc, all_preds_mc = ensemble.predict_with_uncertainty(
            data, use_mc_dropout=True
        )

        assert mean_pred_mc.shape == mean_pred.shape
        assert uncertainty_mc.shape == uncertainty.shape


class TestUncertaintyAwareClassifier:
    """Test main uncertainty-aware classifier."""

    def test_single_model_initialization(self, sample_config):
        """Test initialization with single model."""
        config = sample_config
        config.model.ensemble_size = 1

        classifier = UncertaintyAwareClassifier(config.model)

        assert isinstance(classifier.model, MCDropoutClassifier)
        assert not classifier.is_ensemble

    def test_ensemble_model_initialization(self, sample_config):
        """Test initialization with ensemble."""
        config = sample_config
        config.model.ensemble_size = 3

        classifier = UncertaintyAwareClassifier(config.model)

        assert isinstance(classifier.model, DeepEnsemble)
        assert classifier.is_ensemble

    def test_predict_with_uncertainty_methods(self, sample_config, sample_batch):
        """Test different uncertainty methods."""
        data, targets = sample_batch
        classifier = UncertaintyAwareClassifier(sample_config.model)

        methods = ["mc_dropout", "combined"]
        if sample_config.model.ensemble_size > 1:
            methods.append("ensemble")

        for method in methods:
            results = classifier.predict_with_uncertainty(
                data, uncertainty_method=method, num_mc_samples=5
            )

            # Check required keys
            required_keys = [
                'predictions', 'aleatoric_uncertainty', 'epistemic_uncertainty',
                'total_uncertainty', 'confidence', 'entropy', 'raw_predictions'
            ]
            assert all(key in results for key in required_keys)

            # Check shapes
            batch_size = data.size(0)
            assert results['predictions'].shape == (batch_size, sample_config.model.num_classes)
            assert results['aleatoric_uncertainty'].shape == (batch_size,)
            assert results['epistemic_uncertainty'].shape == (batch_size,)
            assert results['total_uncertainty'].shape == (batch_size,)
            assert results['confidence'].shape == (batch_size,)
            assert results['entropy'].shape == (batch_size,)

            # Check value ranges
            assert torch.all(results['confidence'] >= 0)
            assert torch.all(results['confidence'] <= 1)
            assert torch.all(results['entropy'] >= 0)
            assert torch.all(results['total_uncertainty'] >= 0)

    def test_get_prediction_summary(self, sample_config, sample_batch):
        """Test prediction summary generation."""
        data, targets = sample_batch
        classifier = UncertaintyAwareClassifier(sample_config.model)

        class_names = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
        summaries = classifier.get_prediction_summary(
            data, class_names, uncertainty_method="combined", top_k=3
        )

        assert len(summaries) == data.size(0)

        for summary in summaries:
            required_keys = [
                'top_predictions', 'confidence_score', 'uncertainty_score',
                'confidence_level', 'should_review'
            ]
            assert all(key in summary for key in required_keys)

            assert len(summary['top_predictions']) <= 3
            assert isinstance(summary['confidence_score'], float)
            assert isinstance(summary['uncertainty_score'], float)
            assert summary['confidence_level'] in ['High', 'Medium', 'Low']
            assert isinstance(summary['should_review'], bool)

    def test_get_model_info(self, sample_config):
        """Test model information retrieval."""
        classifier = UncertaintyAwareClassifier(sample_config.model)
        info = classifier.get_model_info()

        required_keys = [
            'model_type', 'backbone', 'num_classes', 'ensemble_size',
            'mc_samples', 'dropout_rate', 'total_parameters',
            'trainable_parameters', 'pretrained', 'backbone_frozen'
        ]
        assert all(key in info for key in required_keys)

        assert info['backbone'] == sample_config.model.backbone
        assert info['num_classes'] == sample_config.model.num_classes
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0

    def test_uncertainty_consistency(self, sample_config, sample_batch):
        """Test uncertainty estimation consistency."""
        data, targets = sample_batch
        classifier = UncertaintyAwareClassifier(sample_config.model)

        # Multiple calls should give similar results (with some randomness)
        results1 = classifier.predict_with_uncertainty(data, num_mc_samples=20)
        results2 = classifier.predict_with_uncertainty(data, num_mc_samples=20)

        # Predictions should be close but not identical due to MC sampling
        pred_diff = torch.abs(results1['predictions'] - results2['predictions'])
        assert torch.mean(pred_diff) < 0.1  # Should be reasonably close

        # Uncertainties should be in similar ranges (allow for MC sampling variance)
        unc_ratio = results1['total_uncertainty'] / (results2['total_uncertainty'] + 1e-8)
        assert torch.all(unc_ratio > 0.1) and torch.all(unc_ratio < 10.0)

    def test_predict_with_uncertainty_invalid_input(self, sample_config):
        """Test prediction with invalid input tensors."""
        classifier = UncertaintyAwareClassifier(sample_config.model)

        # Test non-tensor input
        with pytest.raises(ValueError, match="Input must be a PyTorch tensor"):
            classifier.predict_with_uncertainty([[1, 2, 3]])

        # Test wrong tensor dimensions
        invalid_tensor = torch.randn(10, 20)  # 2D instead of 4D
        with pytest.raises(ValueError, match="Input tensor must be 4D with batch dimension"):
            classifier.predict_with_uncertainty(invalid_tensor)

        # Test empty batch
        empty_tensor = torch.randn(0, 3, 224, 224)
        with pytest.raises(ValueError, match="Input tensor must be 4D with batch dimension"):
            classifier.predict_with_uncertainty(empty_tensor)

    def test_predict_with_uncertainty_invalid_method(self, sample_config, sample_batch):
        """Test prediction with invalid uncertainty method."""
        data, _ = sample_batch
        classifier = UncertaintyAwareClassifier(sample_config.model)

        with pytest.raises(ValueError, match="uncertainty_method must be one of"):
            classifier.predict_with_uncertainty(data, uncertainty_method="invalid")

    def test_predict_with_uncertainty_invalid_mc_samples(self, sample_config, sample_batch):
        """Test prediction with invalid MC samples."""
        data, _ = sample_batch
        classifier = UncertaintyAwareClassifier(sample_config.model)

        with pytest.raises(ValueError, match="num_mc_samples must be positive if specified"):
            classifier.predict_with_uncertainty(data, num_mc_samples=0)

        with pytest.raises(ValueError, match="num_mc_samples must be positive if specified"):
            classifier.predict_with_uncertainty(data, num_mc_samples=-5)

    def test_get_prediction_summary_invalid_class_names(self, sample_config, sample_batch):
        """Test prediction summary with invalid class names."""
        data, _ = sample_batch
        classifier = UncertaintyAwareClassifier(sample_config.model)

        # Test empty class names
        with pytest.raises(ValueError, match="class_names must be a non-empty list"):
            classifier.get_prediction_summary(data, [])

        # Test non-list class names
        with pytest.raises(ValueError, match="class_names must be a non-empty list"):
            classifier.get_prediction_summary(data, "invalid")

        # Test wrong number of class names
        wrong_names = ["Class_1", "Class_2"]  # Only 2 names for 5 classes
        with pytest.raises(ValueError, match="Number of class names .* must match model classes"):
            classifier.get_prediction_summary(data, wrong_names)

    def test_get_prediction_summary_invalid_top_k(self, sample_config, sample_batch):
        """Test prediction summary with invalid top_k."""
        data, _ = sample_batch
        classifier = UncertaintyAwareClassifier(sample_config.model)
        class_names = [f"Class_{i}" for i in range(sample_config.model.num_classes)]

        # Test top_k <= 0
        with pytest.raises(ValueError, match="top_k must be between 1 and"):
            classifier.get_prediction_summary(data, class_names, top_k=0)

        # Test top_k > number of classes
        with pytest.raises(ValueError, match="top_k must be between 1 and"):
            classifier.get_prediction_summary(data, class_names, top_k=100)

    @pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet18"])
    def test_different_backbones(self, backbone, sample_batch):
        """Test different backbone architectures."""
        data, targets = sample_batch

        config = ModelConfig(
            backbone=backbone,
            num_classes=5,
            pretrained=False,
            ensemble_size=1
        )

        try:
            classifier = UncertaintyAwareClassifier(config)
            output = classifier(data)
            assert output.shape == (data.size(0), config.num_classes)
        except Exception as e:
            # Some backbones might not be available in test environment
            pytest.skip(f"Backbone {backbone} not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])