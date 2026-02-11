#!/usr/bin/env python3
"""Inference script for pet breed uncertainty-aware classifier."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pet_breed_uncertainty_aware_classifier.utils.config import load_config
from pet_breed_uncertainty_aware_classifier.models.model import UncertaintyAwareClassifier
from pet_breed_uncertainty_aware_classifier.data.preprocessing import get_transforms


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
) -> tuple:
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


def load_class_names(dataset_path: str = None) -> List[str]:
    """Load class names from dataset or use default Oxford-IIIT Pet classes."""
    # Oxford-IIIT Pet Dataset 37 breed classes
    default_classes = [
        'Abyssinian', 'American_Bulldog', 'American_Pit_Bull_Terrier', 'Basset_Hound',
        'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British_Shorthair',
        'Chihuahua', 'Egyptian_Mau', 'English_Cocker_Spaniel', 'English_Setter',
        'German_Shorthaired', 'Great_Pyrenees', 'Havanese', 'Japanese_Chin',
        'Keeshond', 'Leonberger', 'Maine_Coon', 'Miniature_Pinscher', 'Newfoundland',
        'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian_Blue', 'Saint_Bernard',
        'Samoyed', 'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx',
        'Staffordshire_Bull_Terrier', 'Wheaten_Terrier', 'Yorkshire_Terrier'
    ]

    if dataset_path:
        class_file = Path(dataset_path) / "classes.txt"
        if class_file.exists():
            with open(class_file, 'r') as f:
                return [line.strip() for line in f]

    return default_classes


def load_and_preprocess_image(
    image_path: str,
    transform: transforms.Compose,
    device: torch.device
) -> torch.Tensor:
    """Load and preprocess a single image."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0).to(device)  # Add batch dimension


def predict_single_image(
    model: UncertaintyAwareClassifier,
    image_tensor: torch.Tensor,
    class_names: List[str],
    uncertainty_method: str = "combined",
    top_k: int = 5
) -> Dict[str, Any]:
    """Run inference on a single image."""
    with torch.no_grad():
        # Get predictions with uncertainty
        uncertainty_results = model.predict_with_uncertainty(
            image_tensor,
            uncertainty_method=uncertainty_method
        )

        # Get human-readable summary
        summaries = model.get_prediction_summary(
            image_tensor,
            class_names,
            uncertainty_method=uncertainty_method,
            top_k=top_k
        )

        summary = summaries[0]  # Single image

        # Extract detailed uncertainty information
        predictions = uncertainty_results['predictions'][0]

        return {
            'top_predictions': summary['top_predictions'],
            'confidence_score': summary['confidence_score'],
            'uncertainty_score': summary['uncertainty_score'],
            'confidence_level': summary['confidence_level'],
            'should_review': summary['should_review'],
            'aleatoric_uncertainty': uncertainty_results['aleatoric_uncertainty'][0].item(),
            'epistemic_uncertainty': uncertainty_results['epistemic_uncertainty'][0].item(),
            'predictive_entropy': uncertainty_results['entropy'][0].item()
        }


def predict_batch(
    model: UncertaintyAwareClassifier,
    image_paths: List[str],
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device,
    uncertainty_method: str = "combined",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Run inference on a batch of images."""
    results = []

    for image_path in image_paths:
        try:
            logging.info(f"Processing: {image_path}")

            image_tensor = load_and_preprocess_image(image_path, transform, device)
            prediction = predict_single_image(
                model,
                image_tensor,
                class_names,
                uncertainty_method,
                top_k
            )

            prediction['image_path'] = image_path
            results.append(prediction)

        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            continue

    return results


def format_prediction_output(result: Dict[str, Any]) -> str:
    """Format prediction result for display."""
    output = []
    output.append(f"\nImage: {result['image_path']}")
    output.append("=" * 70)

    output.append("\nTop Predictions:")
    for i, (class_name, prob) in enumerate(result['top_predictions'], 1):
        output.append(f"  {i}. {class_name.replace('_', ' ')}: {prob:.2%}")

    output.append(f"\nConfidence: {result['confidence_score']:.2%} ({result['confidence_level']})")
    output.append(f"Total Uncertainty: {result['uncertainty_score']:.4f}")
    output.append(f"Aleatoric (data) Uncertainty: {result['aleatoric_uncertainty']:.4f}")
    output.append(f"Epistemic (model) Uncertainty: {result['epistemic_uncertainty']:.4f}")
    output.append(f"Predictive Entropy: {result['predictive_entropy']:.4f}")

    if result['should_review']:
        output.append("\n⚠️  LOW CONFIDENCE: Manual review recommended")
    else:
        output.append("\n✓ High confidence prediction")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on pet images with uncertainty quantification"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for inference'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images for batch inference'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output JSON file for predictions (default: predictions.json)'
    )
    parser.add_argument(
        '--uncertainty-method',
        type=str,
        default='combined',
        choices=['mc_dropout', 'ensemble', 'combined'],
        help='Uncertainty estimation method (default: combined)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for inference (default: auto)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate input
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    logger.info("Model loaded successfully")

    # Load class names
    class_names = load_class_names(config.data.dataset_path if hasattr(config, 'data') else None)
    logger.info(f"Loaded {len(class_names)} class names")

    # Get transforms
    transform = get_transforms(
        image_size=config.data.image_size if hasattr(config, 'data') else [224, 224],
        is_training=False
    )

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(str(p) for p in image_dir.glob(ext))

    if not image_paths:
        logger.error("No images found to process")
        return

    logger.info(f"Found {len(image_paths)} images to process")

    # Run inference
    results = predict_batch(
        model,
        image_paths,
        transform,
        class_names,
        device,
        args.uncertainty_method,
        args.top_k
    )

    # Display results
    for result in results:
        print(format_prediction_output(result))

    # Save results to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nPredictions saved to {output_path}")
    logger.info(f"Processed {len(results)} images successfully")


if __name__ == '__main__':
    main()
