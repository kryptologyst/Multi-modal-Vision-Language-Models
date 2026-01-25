"""Evaluation script for Multi-modal Vision-Language Models."""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.models.vl_models import CLIPWrapper, BLIPWrapper
from src.data.datasets import VQADataset, ImageCaptionDataset, create_dataloader
from src.eval.metrics import VLMMetrics, evaluate_model
from src.utils.device import get_device, set_seed
from src.utils.config import Config, load_config


def evaluate_vqa_model(
    model_path: str,
    test_data_path: str,
    image_dir: str,
    config: Config,
    output_dir: str,
) -> Dict[str, float]:
    """Evaluate VQA model on test dataset.
    
    Args:
        model_path: Path to model checkpoint.
        test_data_path: Path to test dataset.
        image_dir: Directory containing images.
        config: Configuration object.
        output_dir: Directory to save results.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    device = get_device()
    set_seed(config.seed)
    
    # Load model
    if config.model.name == "blip":
        model = BLIPWrapper(config.model.pretrained)
    else:
        model = CLIPWrapper(config.model.pretrained)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create dataset
    test_dataset = VQADataset(
        data_path=test_data_path,
        image_dir=image_dir,
        processor=model.processor,
        split="test",
    )
    
    # Create data loader
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    
    # Initialize metrics
    metrics = VLMMetrics()
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        metrics=metrics,
        task="vqa",
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "vqa_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def evaluate_captioning_model(
    model_path: str,
    test_data_path: str,
    image_dir: str,
    config: Config,
    output_dir: str,
) -> Dict[str, float]:
    """Evaluate image captioning model on test dataset.
    
    Args:
        model_path: Path to model checkpoint.
        test_data_path: Path to test dataset.
        image_dir: Directory containing images.
        config: Configuration object.
        output_dir: Directory to save results.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    device = get_device()
    set_seed(config.seed)
    
    # Load model
    if config.model.name == "blip":
        model = BLIPWrapper(config.model.pretrained)
    else:
        model = CLIPWrapper(config.model.pretrained)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create dataset
    test_dataset = ImageCaptionDataset(
        data_path=test_data_path,
        image_dir=image_dir,
        processor=model.processor,
        split="test",
    )
    
    # Create data loader
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    
    # Initialize metrics
    metrics = VLMMetrics()
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        metrics=metrics,
        task="captioning",
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "captioning_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Multi-modal Vision-Language Models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--task", type=str, default="vqa", choices=["vqa", "captioning"], help="Task to evaluate")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"Evaluating {args.task} model...")
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Evaluate based on task
    if args.task == "vqa":
        results = evaluate_vqa_model(
            model_path=args.model_path,
            test_data_path=args.test_data,
            image_dir=args.image_dir,
            config=config,
            output_dir=args.output_dir,
        )
    elif args.task == "captioning":
        results = evaluate_captioning_model(
            model_path=args.model_path,
            test_data_path=args.test_data,
            image_dir=args.image_dir,
            config=config,
            output_dir=args.output_dir,
        )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
