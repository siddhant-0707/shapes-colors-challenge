"""
Training utilities for deep learning models
"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np


def train_epoch(model, train_loader, criterion, optimizer, device, use_amp=False, scaler=None):
    """Train for one epoch with optional AMP support"""
    model.train()
    running_loss = 0.0
    
    for images, targets in tqdm(train_loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            # Use AMP for memory efficiency
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating", leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)


def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001, 
                device='cuda', save_path='best_model.pth', track_jaccard=True, use_amp=False):
    """Complete training loop with early stopping, LR tracking, and Jaccard monitoring
    
    Args:
        use_amp: If True, use Automatic Mixed Precision for memory efficiency
    """
    criterion = nn.BCEWithLogitsLoss()  # More numerically stable than BCE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                            factor=0.5, patience=3)
    
    # Initialize AMP scaler if using mixed precision
    scaler = None
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("   Using Automatic Mixed Precision (AMP) for memory efficiency")
    
    train_losses = []
    val_losses = []
    lr_history = []  # Track learning rate
    jaccard_history = []  # Track Jaccard during training
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Track Jaccard similarity if requested
        if track_jaccard:
            val_jaccard = compute_epoch_jaccard(model, val_loader, device)
            jaccard_history.append(val_jaccard)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Jaccard: {val_jaccard:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(save_path))
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history
    }
    if track_jaccard:
        history['jaccard_history'] = jaccard_history
    
    return history


def compute_epoch_jaccard(model, dataloader, device, threshold=0.5):
    """Compute average Jaccard score on entire dataset during training"""
    from src.metrics import jaccard_similarity
    from src.data import multihot_to_labels
    
    model.eval()
    jaccard_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, targets = batch
            else:
                continue  # Skip if no targets
            
            images = images.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for BCEWithLogitsLoss
            
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.numpy()
            
            for i in range(len(outputs_np)):
                pred_labels = multihot_to_labels(outputs_np[i], threshold)
                true_labels = multihot_to_labels(targets_np[i], threshold)
                score = jaccard_similarity(true_labels, pred_labels)
                jaccard_scores.append(score)
    
    model.train()
    return np.mean(jaccard_scores) if jaccard_scores else 0.0


def predict_with_model(model, dataloader, device, threshold=0.5):
    """Generate predictions using a trained model"""
    from src.data import multihot_to_labels
    import torch.nn.functional as F
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, targets = batch
                targets_np = targets.numpy()
                all_targets.append(targets_np)
            elif isinstance(batch, (tuple, list)) and len(batch) == 1:
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            outputs = model(images)
            # Apply sigmoid since model outputs logits now
            outputs = torch.sigmoid(outputs)
            outputs_np = outputs.cpu().numpy()
            all_predictions.append(outputs_np)
    
    all_predictions = np.vstack(all_predictions)
    pred_labels = [multihot_to_labels(pred, threshold) for pred in all_predictions]
    
    if all_targets:
        all_targets = np.vstack(all_targets)
        true_labels = [multihot_to_labels(target, threshold) for target in all_targets]
        return true_labels, pred_labels
    
    return pred_labels

