"""
Evaluation metrics for multi-label shape-color classification
"""

import numpy as np
from collections import Counter
from src.data import SHAPE_COLOR_PAIRS


def jaccard_similarity(true_labels, pred_labels):
    """Calculate Jaccard similarity between two sets of labels"""
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    
    intersection = len(true_set & pred_set)
    union = len(true_set | pred_set)
    
    return intersection / union if union > 0 else 0.0


def average_jaccard_similarity(true_labels_list, pred_labels_list):
    """Calculate average Jaccard similarity over multiple samples"""
    scores = [jaccard_similarity(true, pred) 
              for true, pred in zip(true_labels_list, pred_labels_list)]
    return np.mean(scores)


def precision_recall_f1(true_labels, pred_labels):
    """Calculate precision, recall, and F1-score for a single prediction"""
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def average_precision_recall_f1(true_labels_list, pred_labels_list):
    """Calculate average precision, recall, and F1 over multiple samples"""
    precisions, recalls, f1s = [], [], []
    
    for true, pred in zip(true_labels_list, pred_labels_list):
        p, r, f = precision_recall_f1(true, pred)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


def exact_match_accuracy(true_labels_list, pred_labels_list):
    """Calculate the percentage of predictions that exactly match the ground truth"""
    exact_matches = sum(1 for true, pred in zip(true_labels_list, pred_labels_list)
                       if set(true) == set(pred))
    return exact_matches / len(true_labels_list)


def per_class_metrics(true_labels_list, pred_labels_list):
    """Calculate precision, recall, and F1-score for each shape-color combination"""
    results = {}
    
    for shape_color in SHAPE_COLOR_PAIRS:
        tp = fp = fn = 0
        
        for true_labels, pred_labels in zip(true_labels_list, pred_labels_list):
            true_has = shape_color in true_labels
            pred_has = shape_color in pred_labels
            
            if true_has and pred_has:
                tp += 1
            elif pred_has and not true_has:
                fp += 1
            elif true_has and not pred_has:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[shape_color] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results


def comprehensive_evaluation(true_labels_list, pred_labels_list, model_name="Model"):
    """Perform comprehensive evaluation of predictions"""
    jaccard = average_jaccard_similarity(true_labels_list, pred_labels_list)
    precision, recall, f1 = average_precision_recall_f1(true_labels_list, pred_labels_list)
    exact_match = exact_match_accuracy(true_labels_list, pred_labels_list)
    per_class = per_class_metrics(true_labels_list, pred_labels_list)
    
    return {
        'model_name': model_name,
        'jaccard': jaccard,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': exact_match,
        'per_class': per_class
    }

