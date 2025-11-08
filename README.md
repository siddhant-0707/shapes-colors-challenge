# Shapes and Colors Prediction Challenge - Solution

## Overview

**Challenge:** Predict geometric shapes (circle, square, triangle) and their colors (red, blue, green) in synthetic images.

## Project Structure

```
shapes-colors/
├── src/                     
│   ├── __init__.py
│   ├── data.py              # Data loading, preprocessing, datasets
│   ├── metrics.py           # Evaluation metrics (Jaccard, F1, etc.)
│   ├── models.py            # Model architectures
│   ├── train.py             # Training utilities
│   ├── traditional_cv.py    # Traditional CV methods
│   └── utils.py             # Visualization and helper functions
├── models/                   # Saved model weights
├── shapes_colors_analysis.ipynb  # Main notebook (insights & analysis)
├── requirements.txt
├── submission.csv
└── README.md               
```
## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Notebook

```bash
jupyter notebook shapes_colors_analysis.ipynb
```

The notebook is concise and focused on insights

## What's in the Notebook?

The notebook (~24 cells) focuses on:

1. **Dataset Exploration** - Visualizations and statistical analysis
2. **Methodology** - Clear explanation of approaches (not implementation)
3. **Model Training** - Orchestrates training using `src/` modules
4. **Results & Analysis** - Performance comparison with visualizations
5. **Key Insights** - Deep analysis of what works and why
6. **Test Predictions** - Generate submission file
7. **Conclusions** - Limitations, future work, and recommendations

## Approaches Implemented

### 1. Traditional Computer Vision (`src/traditional_cv.py`)
- HSV color segmentation
- Contour-based shape detection
- Geometric feature extraction

### 2. Custom CNN (`src/models.py` - `CustomCNN`)
- 4-layer convolutional network
- Batch normalization & dropout
- Multi-label classification

### 3. Transfer Learning (`src/models.py`)
- ResNet18
- EfficientNet-B0
- MobileNetV2

All trained with: Binary Cross Entropy loss, Adam optimizer, early stopping, data augmentation.

<!-- ## Key Modules

### `src/data.py`
- `ShapesColorsDataset`: PyTorch dataset for training
- `TestDataset`: PyTorch dataset for test predictions
- `labels_to_multihot()`: Convert labels to multi-hot encoding
- `multihot_to_labels()`: Convert predictions back to labels
- `create_dataloaders()`: Create train/val dataloaders

### `src/metrics.py`
- `jaccard_similarity()`: Primary metric
- `precision_recall_f1()`: Standard multi-label metrics
- `comprehensive_evaluation()`: Complete evaluation suite
- `per_class_metrics()`: Per shape-color performance

### `src/models.py`
- `CustomCNN`: CNN from scratch
- `create_resnet18()`: ResNet18 for multi-label
- `create_efficientnet()`: EfficientNet-B0 for multi-label
- `create_mobilenet()`: MobileNetV2 for multi-label

### `src/train.py`
- `train_model()`: Complete training loop with early stopping
- `predict_with_model()`: Generate predictions

### `src/traditional_cv.py`
- `detect_color()`: Color segmentation
- `classify_shape()`: Shape classification from contours
- `traditional_cv_predict()`: Complete CV pipeline

### `src/utils.py`
- `visualize_samples()`: Display images with labels
- `plot_distributions()`: Data distribution plots
- `plot_model_comparison()`: Model comparison charts
- `plot_predictions()`: Predictions with Jaccard scores -->

## Running the Full Pipeline

```python
# In the notebook or a script:
from src.data import create_dataloaders
from src.models import create_resnet18
from src.train import train_model, predict_with_model
from src.metrics import comprehensive_evaluation

# Load data (see notebook for details)
train_loader, val_loader = create_dataloaders(train_data, val_data, data_dir)

# Train model
model = create_resnet18(pretrained=True).to(device)
train_losses, val_losses = train_model(model, train_loader, val_loader)

# Evaluate
true_labels, pred_labels = predict_with_model(model, val_loader, device)
results = comprehensive_evaluation(true_labels, pred_labels, "ResNet18")
```

## Key Results

Transfer learning models achieve the best performance:
- **Best Jaccard Similarity**: ~0.85-0.95 (depending on model)
- **Transfer Learning >> Custom CNN >> Traditional CV**
- ResNet18/EfficientNet-B0 recommended for production

See the notebook for detailed analysis and insights.

## Author
Siddhant Chauhan

University of Florida
