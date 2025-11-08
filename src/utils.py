"""
Utility functions for visualization and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from src.metrics import jaccard_similarity


def visualize_samples(dataframe, data_dir, num_samples=15, title="Sample Images"):
    """Visualize sample images with their labels"""
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    axes = axes.flatten()
    
    indices = np.linspace(0, len(dataframe)-1, num_samples, dtype=int)
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
        
        img_path = data_dir / dataframe.iloc[idx]['image_path']
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        labels = dataframe.iloc[idx]['parsed_labels']
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Image {idx}\n{labels}", fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_distributions(dataframe):
    """Plot data distributions"""
    from collections import Counter
    from src.data import SHAPES, COLORS, SHAPE_COLOR_PAIRS
    
    # Gather statistics
    all_shapes = []
    all_colors = []
    for labels in dataframe['parsed_labels']:
        for shape, color in labels:
            all_shapes.append(shape)
            all_colors.append(color)
    
    shape_counter = Counter(all_shapes)
    color_counter = Counter(all_colors)
    pair_counter = Counter([(s, c) for labels in dataframe['parsed_labels'] for s, c in labels])
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Shape distribution
    shape_df = pd.DataFrame.from_dict(shape_counter, orient='index', columns=['count']).sort_index()
    shape_df.plot(kind='bar', ax=axes[0, 0], legend=False, color='coral')
    axes[0, 0].set_title('Shape Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Shape')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Color distribution
    color_df = pd.DataFrame.from_dict(color_counter, orient='index', columns=['count']).sort_index()
    colors_map = {'blue': 'blue', 'green': 'green', 'red': 'red'}
    color_list = [colors_map[c] for c in color_df.index]
    color_df.plot(kind='bar', ax=axes[0, 1], legend=False, color=color_list)
    axes[0, 1].set_title('Color Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Color')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Shape-Color heatmap
    pair_matrix = np.zeros((len(SHAPES), len(COLORS)))
    for i, shape in enumerate(SHAPES):
        for j, color in enumerate(COLORS):
            pair_matrix[i, j] = pair_counter.get((shape, color), 0)
    
    sns.heatmap(pair_matrix, annot=True, fmt='g', xticklabels=COLORS, yticklabels=SHAPES,
                cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Count'})
    axes[1, 0].set_title('Shape-Color Combination Heatmap', fontsize=12, fontweight='bold')
    
    # Object count distribution
    dataframe['num_objects'] = dataframe['parsed_labels'].apply(len)
    dataframe['num_objects'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1], color='skyblue')
    axes[1, 1].set_title('Objects per Image Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Objects')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(results_dict):
    """Plot comparison of multiple models"""
    results_df = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['jaccard', 'precision', 'recall', 'f1']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(results_df.index, results_df[metric], color=colors)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(results_df.index, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_predictions(dataframe, data_dir, predictions, true_labels, num_samples=12, title="Predictions"):
    """Visualize predictions with Jaccard scores"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataframe))):
        img_path = data_dir / dataframe.iloc[i]['image_path']
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        true = true_labels[i]
        pred = predictions[i]
        score = jaccard_similarity(true, pred)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"J={score:.2f}\nTrue: {true}\nPred: {pred}", fontsize=8)
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def print_results_table(results_dict):
    """Print formatted results table"""
    df = pd.DataFrame({
        'Jaccard': {k: v['jaccard'] for k, v in results_dict.items()},
        'Precision': {k: v['precision'] for k, v in results_dict.items()},
        'Recall': {k: v['recall'] for k, v in results_dict.items()},
        'F1-Score': {k: v['f1'] for k, v in results_dict.items()},
        'Exact Match': {k: v['exact_match'] for k, v in results_dict.items()},
    })
    
    print("="*80)
    print("MODEL COMPARISON - ALL METRICS")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    best_model = df['Jaccard'].idxmax()
    print(f"\nBest Model (by Jaccard Similarity): {best_model}")
    print(f"Jaccard Score: {df.loc[best_model, 'Jaccard']:.4f}")
    
    return df


def plot_training_curves(history_dict, title="Training History"):
    """Plot training curves for multiple models
    
    Args:
        history_dict: Dict of {model_name: {'train_losses': [...], 'val_losses': [...], 'lr_history': [...]}}
    """
    n_models = len(history_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, history) in enumerate(history_dict.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        epochs = range(1, len(history['train_losses']) + 1)
        
        ax.plot(epochs, history['train_losses'], 'o-', label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_losses'], 's-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_learning_rates(history_dict, title="Learning Rate Schedule"):
    """Plot learning rate history for all models
    
    Args:
        history_dict: Dict of {model_name: {'train_losses': [...], 'val_losses': [...], 'lr_history': [...]}}
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    for model_name, history in history_dict.items():
        if 'lr_history' in history and len(history['lr_history']) > 0:
            epochs = range(1, len(history['lr_history']) + 1)
            ax.plot(epochs, history['lr_history'], 'o-', label=model_name, linewidth=2)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    return fig


def plot_jaccard_history(history_dict, title="Jaccard Similarity During Training"):
    """Plot Jaccard similarity evolution during training
    
    Args:
        history_dict: Dict of {model_name: {'train_losses': [...], 'jaccard_history': [...]}}
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    for model_name, history in history_dict.items():
        if 'jaccard_history' in history and len(history['jaccard_history']) > 0:
            epochs = range(1, len(history['jaccard_history']) + 1)
            ax.plot(epochs, history['jaccard_history'], 'o-', label=model_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Jaccard Similarity', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])  # Jaccard is between 0 and 1
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_highest_loss_cases(model, dataloader, criterion, device, dataframe, 
                                   data_dir, num_cases=6):
    """Visualize samples with highest validation loss to understand failure modes"""
    from src.data import multihot_to_labels
    import torch
    
    model.eval()
    sample_data = []
    
    with torch.no_grad():
        batch_idx = 0
        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, targets = batch
            else:
                continue
            
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            # Compute loss per sample
            for i in range(len(outputs)):
                loss = criterion(outputs[i:i+1], targets[i:i+1])
                data_idx = batch_idx * dataloader.batch_size + i
                
                if data_idx < len(dataframe):
                    sample_data.append({
                        'index': data_idx,
                        'loss': loss.item(),
                        'pred_vector': torch.sigmoid(outputs[i]).cpu().numpy(),
                        'true_vector': targets[i].cpu().numpy()
                    })
            
            batch_idx += 1
            if batch_idx * dataloader.batch_size >= len(dataframe):
                break
    
    # Sort by loss and get worst cases
    sample_data.sort(key=lambda x: x['loss'], reverse=True)
    worst_cases = sample_data[:num_cases]
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, case in enumerate(worst_cases):
        if idx >= len(axes):
            break
        
        row = dataframe.iloc[case['index']]
        img_path = os.path.join(data_dir, row['image_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        true_labels = multihot_to_labels(case['true_vector'])
        pred_labels = multihot_to_labels(case['pred_vector'])
        
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"Loss: {case['loss']:.4f}\n"
            f"True: {true_labels}\n"
            f"Pred: {pred_labels}",
            fontsize=9
        )
        axes[idx].axis('off')
    
    plt.suptitle("Highest Loss Cases - Model's Biggest Challenges", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    model.train()
    return fig


def plot_error_analysis(dataframe, data_dir, predictions, true_labels, num_samples=18):
    """Categorize and visualize predictions by performance"""
    from src.metrics import jaccard_similarity
    
    scores = [jaccard_similarity(true, pred) for true, pred in zip(true_labels, predictions)]
    
    # Categorize
    perfect = [i for i, s in enumerate(scores) if s == 1.0]
    good = [i for i, s in enumerate(scores) if 0.5 < s < 1.0]
    poor = [i for i, s in enumerate(scores) if s <= 0.5]
    
    print(f"Performance Distribution:")
    print(f"  Perfect (J=1.0): {len(perfect)} ({len(perfect)/len(scores)*100:.1f}%)")
    print(f"  Good (0.5<J<1.0): {len(good)} ({len(good)/len(scores)*100:.1f}%)")
    print(f"  Poor (J≤0.5): {len(poor)} ({len(poor)/len(scores)*100:.1f}%)")
    
    # Visualize examples from each category
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    
    categories = [
        (perfect[:6], 'Perfect Predictions (Jaccard=1.0)', 0),
        (good[:6], 'Good Predictions (0.5<Jaccard<1.0)', 1),
        (poor[:6], 'Poor Predictions (Jaccard≤0.5)', 2)
    ]
    
    for indices, title, row in categories:
        for col, idx in enumerate(indices):
            if col >= 6 or idx >= len(dataframe):
                axes[row, col].axis('off')
                continue
            
            img_path = data_dir / dataframe.iloc[idx]['image_path']
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            true = true_labels[idx]
            pred = predictions[idx]
            score = jaccard_similarity(true, pred)
            
            axes[row, col].imshow(img_rgb)
            axes[row, col].set_title(f"J={score:.2f}\nTrue:{true}\nPred:{pred}", fontsize=7)
            axes[row, col].axis('off')
        
        # Add category label
        if indices:
            axes[row, 0].text(-0.1, 0.5, title, transform=axes[row, 0].transAxes,
                            fontsize=11, fontweight='bold', rotation=90, va='center')
    
    plt.tight_layout()
    return fig

