"""
Traditional computer vision methods for shape and color detection
"""

import cv2
import numpy as np
from src.data import COLORS, SHAPES


# HSV color ranges for detection
COLOR_RANGES = {
    'red': [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
    ],
    'green': [
        (np.array([40, 50, 50]), np.array([80, 255, 255]))
    ],
    'blue': [
        (np.array([100, 100, 100]), np.array([130, 255, 255]))
    ]
}


def detect_color(image, color_name):
    """Detect regions of a specific color in the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for lower, upper in COLOR_RANGES[color_name]:
        color_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, color_mask)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def classify_shape(contour):
    """Classify shape based on contour properties"""
    area = cv2.contourArea(contour)
    
    if area < 100:
        return None
    
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    vertices = len(approx)
    
    if circularity > 0.75:
        return 'circle'
    elif vertices == 3:
        return 'triangle'
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 0
        if 0.7 <= aspect_ratio <= 1.3:
            return 'square'
    elif vertices >= 5:
        return 'circle'
    
    return None


def detect_shapes_in_mask(mask):
    """Detect and classify shapes in a binary mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_shapes = []
    for contour in contours:
        shape = classify_shape(contour)
        if shape is not None:
            detected_shapes.append((shape, contour))
    
    return detected_shapes


def traditional_cv_predict(image_path):
    """Complete traditional CV pipeline to predict shapes and colors"""
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    detected = []
    for color in COLORS:
        mask = detect_color(img_rgb, color)
        shapes = detect_shapes_in_mask(mask)
        
        for shape_name, _ in shapes:
            detected.append((shape_name, color))
    
    return list(set(detected))


def evaluate_traditional_cv(dataframe, data_dir):
    """Evaluate traditional CV approach on a dataset"""
    from tqdm.auto import tqdm
    
    predictions = []
    true_labels = []
    
    for idx in tqdm(range(len(dataframe)), desc="Traditional CV predictions"):
        img_path = data_dir / dataframe.iloc[idx]['image_path']
        pred = traditional_cv_predict(img_path)
        true = dataframe.iloc[idx]['parsed_labels']
        
        predictions.append(pred)
        true_labels.append(true)
    
    return true_labels, predictions

