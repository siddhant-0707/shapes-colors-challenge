#!/usr/bin/env python3
"""
Standalone training script for Shapes and Colors Prediction Challenge.

This script trains all models (CustomCNN, ResNet18, EfficientNet-B0, MobileNetV2)
and saves the trained models and training history for use in the analysis notebook.

Memory-efficient features:
- Automatic CUDA cache clearing between models
- Smaller batch size for EfficientNet (default: 16 vs 32)
- Optional Automatic Mixed Precision (AMP) support
- Per-model dataloader creation to free memory

Usage:
    # Basic usage
    python train.py --train_csv dataset_v3/train.csv --train_dir dataset_v3 --num_epochs 15 --batch_size 32 --output_dir models
    
    # Memory-efficient (recommended for GPUs with <8GB VRAM)
    python train.py --train_csv dataset_v3/train.csv --train_dir dataset_v3 --num_epochs 15 --batch_size 16 --batch_size_large 8 --use_amp --output_dir models
    
    # Train only EfficientNet with memory optimizations
    python train.py --train_csv dataset_v3/train.csv --train_dir dataset_v3 --models efficientnet --batch_size_large 8 --use_amp --output_dir models
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import random

from src.data import parse_label, create_dataloaders
from src.models import CustomCNN, create_resnet18, create_efficientnet, create_mobilenet
from src.train import train_model, predict_with_model
from src.metrics import comprehensive_evaluation
from src.traditional_cv import evaluate_traditional_cv


def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train models for Shapes and Colors Prediction')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for smaller models (default: 32)')
    parser.add_argument('--batch_size_large', type=int, default=16,
                        help='Batch size for larger models like EfficientNet (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for custom CNN (default: 0.001)')
    parser.add_argument('--transfer_lr', type=float, default=0.0001,
                        help='Learning rate for transfer learning models (default: 0.0001)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4, use 0 for Jupyter)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP) for memory efficiency')
    parser.add_argument('--no_clear_cache', action='store_true',
                        help='Do not clear CUDA cache between models (clearing is default for memory efficiency)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models and results (default: models)')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['cnn', 'resnet18', 'efficientnet', 'mobilenet'],
                        choices=['cnn', 'resnet18', 'efficientnet', 'mobilenet', 'all', 'traditional_cv'],
                        help='Models to train (default: all deep learning models)')
    
    args = parser.parse_args()
    
    # Set seeds
    set_seeds(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    train_csv = Path(args.train_csv)
    train_dir = Path(args.train_dir)
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    train_df = pd.read_csv(train_csv)
    train_df['parsed_labels'] = train_df['label'].apply(parse_label)
    
    # Split data
    train_data, val_data = train_test_split(
        train_df, test_size=args.val_split, random_state=42
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Store results
    results = {}
    training_history = {}
    
    # Helper function to clear GPU cache
    def clear_gpu_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    # 1. Traditional CV (if requested)
    if 'traditional_cv' in args.models or 'all' in args.models:
        print("\n1. Traditional Computer Vision...")
        val_subset = val_data.head(200)  # Use subset for speed
        true_cv, pred_cv = evaluate_traditional_cv(val_subset, train_dir)
        results['Traditional CV'] = comprehensive_evaluation(true_cv, pred_cv, "Traditional CV")
        print(f"   Jaccard: {results['Traditional CV']['jaccard']:.4f}")
    
    # 2. Custom CNN
    if 'cnn' in args.models or 'all' in args.models:
        if not args.no_clear_cache:
            clear_gpu_cache()
        print("\n2. Custom CNN from Scratch...")
        train_loader, val_loader = create_dataloaders(
            train_data, val_data, train_dir, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        model_cnn = CustomCNN().to(device)
        save_path = output_dir / 'best_cnn.pth'
        history_cnn = train_model(
            model_cnn, train_loader, val_loader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device,
            save_path=str(save_path),
            track_jaccard=True,
            use_amp=args.use_amp
        )
        training_history['Custom CNN'] = history_cnn
        
        # Evaluate
        true_cnn, pred_cnn = predict_with_model(model_cnn, val_loader, device)
        results['Custom CNN'] = comprehensive_evaluation(true_cnn, pred_cnn, "Custom CNN")
        print(f"   Jaccard: {results['Custom CNN']['jaccard']:.4f}")
        del model_cnn
        if not args.no_clear_cache:
            clear_gpu_cache()
    
    # 3. ResNet18
    if 'resnet18' in args.models or 'all' in args.models:
        if not args.no_clear_cache:
            clear_gpu_cache()
        print("\n3. ResNet18 Transfer Learning...")
        train_loader, val_loader = create_dataloaders(
            train_data, val_data, train_dir, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        model_resnet = create_resnet18(pretrained=True).to(device)
        save_path = output_dir / 'best_resnet.pth'
        history_res = train_model(
            model_resnet, train_loader, val_loader,
            num_epochs=args.num_epochs,
            lr=args.transfer_lr,
            device=device,
            save_path=str(save_path),
            track_jaccard=True,
            use_amp=args.use_amp
        )
        training_history['ResNet18'] = history_res
        
        # Evaluate
        true_res, pred_res = predict_with_model(model_resnet, val_loader, device)
        results['ResNet18'] = comprehensive_evaluation(true_res, pred_res, "ResNet18")
        print(f"   Jaccard: {results['ResNet18']['jaccard']:.4f}")
        del model_resnet
        if not args.no_clear_cache:
            clear_gpu_cache()
    
    # 4. EfficientNet-B0 (uses smaller batch size due to memory requirements)
    if 'efficientnet' in args.models or 'all' in args.models:
        if not args.no_clear_cache:
            clear_gpu_cache()
        print("\n4. EfficientNet-B0 Transfer Learning...")
        print(f"   Using batch size: {args.batch_size_large} (memory-efficient)")
        train_loader_eff, val_loader_eff = create_dataloaders(
            train_data, val_data, train_dir, 
            batch_size=args.batch_size_large, 
            num_workers=args.num_workers
        )
        model_eff = create_efficientnet(pretrained=True).to(device)
        save_path = output_dir / 'best_efficientnet.pth'
        history_eff = train_model(
            model_eff, train_loader_eff, val_loader_eff,
            num_epochs=args.num_epochs,
            lr=args.transfer_lr,
            device=device,
            save_path=str(save_path),
            track_jaccard=True,
            use_amp=args.use_amp
        )
        training_history['EfficientNet-B0'] = history_eff
        
        # Evaluate
        true_eff, pred_eff = predict_with_model(model_eff, val_loader_eff, device)
        results['EfficientNet-B0'] = comprehensive_evaluation(true_eff, pred_eff, "EfficientNet-B0")
        print(f"   Jaccard: {results['EfficientNet-B0']['jaccard']:.4f}")
        del model_eff, train_loader_eff, val_loader_eff
        if not args.no_clear_cache:
            clear_gpu_cache()
    
    # 5. MobileNetV2
    if 'mobilenet' in args.models or 'all' in args.models:
        if not args.no_clear_cache:
            clear_gpu_cache()
        print("\n5. MobileNetV2 Transfer Learning...")
        train_loader, val_loader = create_dataloaders(
            train_data, val_data, train_dir, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        model_mobile = create_mobilenet(pretrained=True).to(device)
        save_path = output_dir / 'best_mobilenet.pth'
        history_mob = train_model(
            model_mobile, train_loader, val_loader,
            num_epochs=args.num_epochs,
            lr=args.transfer_lr,
            device=device,
            save_path=str(save_path),
            track_jaccard=True,
            use_amp=args.use_amp
        )
        training_history['MobileNetV2'] = history_mob
        
        # Evaluate
        true_mob, pred_mob = predict_with_model(model_mobile, val_loader, device)
        results['MobileNetV2'] = comprehensive_evaluation(true_mob, pred_mob, "MobileNetV2")
        print(f"   Jaccard: {results['MobileNetV2']['jaccard']:.4f}")
        del model_mobile
        if not args.no_clear_cache:
            clear_gpu_cache()
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save training history (convert numpy arrays to lists for JSON)
    history_save = {}
    for model_name, history in training_history.items():
        history_save[model_name] = {
            'train_losses': [float(x) for x in history['train_losses']],
            'val_losses': [float(x) for x in history['val_losses']],
            'lr_history': [float(x) for x in history['lr_history']],
            'jaccard_history': [float(x) for x in history.get('jaccard_history', [])]
        }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history_save, f, indent=2)
    
    # Save evaluation results
    results_save = {}
    for model_name, result in results.items():
        results_save[model_name] = {
            'jaccard': float(result['jaccard']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1': float(result['f1']),
            'exact_match': float(result['exact_match'])
        }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_save, f, indent=2)
    
    print(f"\n✅ Models saved to: {output_dir}")
    print(f"✅ Training history saved to: {output_dir / 'training_history.json'}")
    print(f"✅ Evaluation results saved to: {output_dir / 'results.json'}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    for model_name, result in results.items():
        print(f"{model_name:20s} - Jaccard: {result['jaccard']:.4f}, F1: {result['f1']:.4f}")
    
    print("\n✅ All models trained successfully!")
    print(f"📊 You can now run the notebook to see analysis and visualizations.")


if __name__ == '__main__':
    main()

