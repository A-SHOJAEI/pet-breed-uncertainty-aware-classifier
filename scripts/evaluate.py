#!/usr/bin/env python3
"""Evaluation script for pet breed uncertainty-aware classifier."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pet_breed_uncertainty_aware_classifier.utils.config import load_config
from pet_breed_uncertainty_aware_classifier.data.loader import PetDataLoader
from pet_breed_uncertainty_aware_classifier.data.preprocessing import get_transforms
from pet_breed_uncertainty_aware_classifier.models.model import UncertaintyAwareClassifier
from pet_breed_uncertainty_aware_classifier.evaluation.metrics import (
    UncertaintyMetrics, ReliabilityDiagram
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> UncertaintyAwareClassifier:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Fallback: try to load config from same directory
        config_path = Path(checkpoint_path).parent.parent / "final_config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            raise ValueError("No config found in checkpoint or adjacent files")

    model = UncertaintyAwareClassifier(config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def evaluate_model(
    model: UncertaintyAwareClassifier,
    data_loader,
    device: torch.device,
    uncertainty_method: str = "combined"
) -> Dict[str, Any]:
    """Evaluate model on given data loader."""
    model.eval()

    all_predictions = []
    all_uncertainties = []
    all_confidences = []
    all_targets = []
    all_prediction_summaries = []

    class_names = data_loader.dataset.CLASS_NAMES

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)

            # Get predictions with uncertainty
            uncertainty_results = model.predict_with_uncertainty(
                data, uncertainty_method=uncertainty_method
            )

            predictions = uncertainty_results['predictions']
            uncertainties = uncertainty_results['total_uncertainty']
            confidences = uncertainty_results['confidence']

            # Get prediction summaries
            summaries = model.get_prediction_summary(
                data, class_names, uncertainty_method=uncertainty_method, top_k=3
            )

            # Collect results
            all_predictions.append(predictions.cpu())
            all_uncertainties.append(uncertainties.cpu())
            all_confidences.append(confidences.cpu())
            all_targets.append(targets.cpu())
            all_prediction_summaries.extend(summaries)

            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    all_confidences = torch.cat(all_confidences, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return {
        'predictions': all_predictions,
        'uncertainties': all_uncertainties,
        'confidences': all_confidences,
        'targets': all_targets,
        'prediction_summaries': all_prediction_summaries,
        'class_names': class_names
    }


def plot_uncertainty_analysis(
    results: Dict[str, Any],
    save_dir: Path
) -> None:
    """Generate uncertainty analysis plots."""
    predictions = results['predictions'].numpy()
    uncertainties = results['uncertainties'].numpy()
    confidences = results['confidences'].numpy()
    targets = results['targets'].numpy()

    # Predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    correct_mask = (predicted_classes == targets)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Uncertainty Analysis', fontsize=16)

    # Plot 1: Uncertainty vs Confidence
    axes[0, 0].scatter(
        confidences[correct_mask], uncertainties[correct_mask],
        alpha=0.6, c='green', label='Correct', s=20
    )
    axes[0, 0].scatter(
        confidences[~correct_mask], uncertainties[~correct_mask],
        alpha=0.6, c='red', label='Incorrect', s=20
    )
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Uncertainty')
    axes[0, 0].set_title('Uncertainty vs Confidence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Confidence distribution
    axes[0, 1].hist(
        confidences[correct_mask], bins=50, alpha=0.7,
        color='green', label='Correct', density=True
    )
    axes[0, 1].hist(
        confidences[~correct_mask], bins=50, alpha=0.7,
        color='red', label='Incorrect', density=True
    )
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].legend()

    # Plot 3: Uncertainty distribution
    axes[1, 0].hist(
        uncertainties[correct_mask], bins=50, alpha=0.7,
        color='green', label='Correct', density=True
    )
    axes[1, 0].hist(
        uncertainties[~correct_mask], bins=50, alpha=0.7,
        color='red', label='Incorrect', density=True
    )
    axes[1, 0].set_xlabel('Uncertainty')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Uncertainty Distribution')
    axes[1, 0].legend()

    # Plot 4: Accuracy vs Confidence bins
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_centers = []
    bin_counts = []

    for i in range(len(confidence_bins) - 1):
        lower = confidence_bins[i]
        upper = confidence_bins[i + 1]
        mask = (confidences >= lower) & (confidences < upper)

        if np.sum(mask) > 0:
            accuracy = np.mean(correct_mask[mask])
            bin_accuracies.append(accuracy)
            bin_centers.append((lower + upper) / 2)
            bin_counts.append(np.sum(mask))

    axes[1, 1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='skyblue')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Calibration Plot')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    results: Dict[str, Any],
    save_dir: Path,
    normalize: bool = True
) -> None:
    """Plot confusion matrix."""
    predictions = results['predictions'].numpy()
    targets = results['targets'].numpy()
    class_names = results['class_names']

    predicted_classes = np.argmax(predictions, axis=1)

    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, predicted_classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'

    # Plot
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=False, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_low_confidence_predictions(
    results: Dict[str, Any],
    save_dir: Path,
    confidence_threshold: float = 0.5
) -> None:
    """Analyze predictions with low confidence."""
    confidences = results['confidences'].numpy()
    uncertainties = results['uncertainties'].numpy()
    summaries = results['prediction_summaries']

    # Find low confidence predictions
    low_conf_mask = confidences < confidence_threshold
    low_conf_indices = np.where(low_conf_mask)[0]

    if len(low_conf_indices) == 0:
        print(f"No predictions found with confidence < {confidence_threshold}")
        return

    # Analyze low confidence cases
    analysis = {
        'total_low_confidence': len(low_conf_indices),
        'percentage_low_confidence': len(low_conf_indices) / len(confidences) * 100,
        'avg_confidence': float(np.mean(confidences[low_conf_mask])),
        'avg_uncertainty': float(np.mean(uncertainties[low_conf_mask])),
        'examples': []
    }

    # Sample some examples
    sample_indices = np.random.choice(
        low_conf_indices,
        min(20, len(low_conf_indices)),
        replace=False
    )

    for idx in sample_indices:
        summary = summaries[idx]
        analysis['examples'].append({
            'index': int(idx),
            'confidence': float(confidences[idx]),
            'uncertainty': float(uncertainties[idx]),
            'top_predictions': summary['top_predictions'],
            'should_review': summary['should_review']
        })

    # Save analysis
    with open(save_dir / 'low_confidence_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Low confidence analysis saved to {save_dir / 'low_confidence_analysis.json'}")
    print(f"Found {analysis['total_low_confidence']} predictions "
          f"({analysis['percentage_low_confidence']:.1f}%) with confidence < {confidence_threshold}")


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty-aware pet breed classifier"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Data split to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--uncertainty-method",
        type=str,
        default="combined",
        choices=["mc_dropout", "ensemble", "combined"],
        help="Uncertainty estimation method"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Evaluation device"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save analysis plots"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Threshold for low-confidence analysis"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting evaluation of uncertainty-aware classifier")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data split: {args.data_split}")

    try:
        # Setup device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        logger.info(f"Using device: {device}")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info("Loading model from checkpoint...")
        model, config = load_model_from_checkpoint(args.checkpoint, device)
        logger.info("Model loaded successfully")

        # Setup data loader
        logger.info("Setting up data loader...")
        data_loader = PetDataLoader(config.data)

        val_transform = get_transforms(
            image_size=config.data.image_size,
            augmentation_strength=0.0,
            is_training=False
        )

        data_loader.prepare_datasets(val_transform, val_transform)

        if args.data_split == "test":
            eval_loader = data_loader.get_test_loader()
        else:
            eval_loader = data_loader.get_val_loader()

        logger.info(f"Evaluating on {len(eval_loader.dataset)} samples")

        # Run evaluation
        logger.info(f"Running evaluation with {args.uncertainty_method} uncertainty method...")
        results = evaluate_model(model, eval_loader, device, args.uncertainty_method)

        # Compute comprehensive metrics
        logger.info("Computing metrics...")
        metrics_calculator = UncertaintyMetrics()
        metrics = metrics_calculator.compute_all_metrics(
            predictions=results['predictions'],
            targets=results['targets'],
            uncertainties=results['uncertainties'],
            confidences=results['confidences'],
            class_names=results['class_names']
        )

        # Print metrics summary
        metrics_calculator.print_summary(metrics)

        # Save metrics to file
        metrics_file = output_dir / 'metrics.json'

        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_metrics[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        serializable_metrics[key][subkey] = subvalue.tolist()
                    else:
                        serializable_metrics[key][subkey] = subvalue
            else:
                serializable_metrics[key] = float(value) if isinstance(value, np.floating) else value

        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_file}")

        # Generate plots if requested
        if args.save_plots:
            logger.info("Generating analysis plots...")

            plot_uncertainty_analysis(results, output_dir)
            plot_confusion_matrix(results, output_dir, normalize=True)
            plot_confusion_matrix(results, output_dir, normalize=False)

            # Generate reliability diagram
            from pet_breed_uncertainty_aware_classifier.evaluation.metrics import CalibrationError
            calibration_error = CalibrationError()
            reliability_diagram = ReliabilityDiagram(calibration_error)

            fig = reliability_diagram.plot(
                results['predictions'],
                results['targets'],
                results['confidences'],
                save_path=str(output_dir / 'reliability_diagram.png')
            )
            plt.close(fig)

            logger.info(f"Plots saved to {output_dir}")

        # Analyze low confidence predictions
        logger.info("Analyzing low confidence predictions...")
        analyze_low_confidence_predictions(
            results, output_dir, args.confidence_threshold
        )

        # Save prediction summaries sample
        sample_summaries = results['prediction_summaries'][:100]  # First 100 samples
        with open(output_dir / 'sample_predictions.json', 'w') as f:
            json.dump(sample_summaries, f, indent=2)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()