"""
Data preprocessing for breast cancer image classification
"""

import os
try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class ImageDataPreprocessor:
    """Preprocessor for breast cancer image dataset."""
    
    def __init__(self, dataset_path: str, image_size: Tuple[int, int] = (224, 224)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_images_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess images with labels."""
        images = []
        labels = []
        
        # Process benign images
        benign_path = os.path.join(self.dataset_path, 'benign')
        for filename in os.listdir(benign_path):
            if filename.endswith('.png') and not filename.endswith('_mask.png'):
                img_path = os.path.join(benign_path, filename)
                img = self._preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append('benign')
        
        # Process malignant images  
        malignant_path = os.path.join(self.dataset_path, 'malignant')
        for filename in os.listdir(malignant_path):
            if filename.endswith('.png') and not filename.endswith('_mask.png'):
                img_path = os.path.join(malignant_path, filename)
                img = self._preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append('malignant')
        
        logger.info(f"Loaded {len(images)} images: {labels.count('benign')} benign, {labels.count('malignant')} malignant")
        
        return np.array(images), np.array(labels)
    
    def _preprocess_image(self, img_path: str) -> np.ndarray:
        """Preprocess individual image."""
        try:
            if cv2 is not None:
                # Use OpenCV if available
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None
                img = cv2.resize(img, self.image_size)
                img = img.astype(np.float32) / 255.0
                img = cv2.equalizeHist((img * 255).astype(np.uint8)).astype(np.float32) / 255.0
            else:
                # Use PIL as fallback
                img = Image.open(img_path).convert('L')
                img = img.resize(self.image_size)
                img = np.array(img, dtype=np.float32) / 255.0
            
            return img.flatten()  # Flatten for traditional ML
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return None
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract enhanced statistical and texture features from images."""
        features = []
        
        for img in images:
            img_2d = img.reshape(self.image_size)
            
            # Enhanced statistical features
            feature_vector = [
                np.mean(img_2d),           # Mean intensity
                np.std(img_2d),            # Standard deviation
                np.var(img_2d),            # Variance
                np.min(img_2d),            # Minimum intensity
                np.max(img_2d),            # Maximum intensity
                np.median(img_2d),         # Median intensity
                np.percentile(img_2d, 10), # 10th percentile
                np.percentile(img_2d, 25), # 25th percentile
                np.percentile(img_2d, 75), # 75th percentile
                np.percentile(img_2d, 90), # 90th percentile
                np.ptp(img_2d),            # Peak-to-peak (range)
                np.mean(img_2d**2),        # Mean of squares
            ]
            
            # Texture features (enhanced)
            if cv2 is not None:
                sobel_x = cv2.Sobel(img_2d, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img_2d, cv2.CV_64F, 0, 1, ksize=3)
            else:
                # Use numpy gradient as fallback
                sobel_x = np.gradient(img_2d, axis=1)
                sobel_y = np.gradient(img_2d, axis=0)
            
            # Enhanced edge features
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            feature_vector.extend([
                np.mean(np.abs(sobel_x)),      # Edge strength X
                np.mean(np.abs(sobel_y)),      # Edge strength Y
                np.std(sobel_x),               # Edge variation X
                np.std(sobel_y),               # Edge variation Y
                np.mean(gradient_magnitude),   # Gradient magnitude mean
                np.std(gradient_magnitude),    # Gradient magnitude std
                np.max(gradient_magnitude),    # Max gradient
                np.percentile(gradient_magnitude, 95), # 95th percentile gradient
            ])
            
            # Local Binary Pattern-like features (simplified)
            center = img_2d[1:-1, 1:-1]
            neighbors = [
                img_2d[:-2, :-2], img_2d[:-2, 1:-1], img_2d[:-2, 2:],
                img_2d[1:-1, :-2],                    img_2d[1:-1, 2:],
                img_2d[2:, :-2],   img_2d[2:, 1:-1],   img_2d[2:, 2:]
            ]
            
            lbp_features = []
            for neighbor in neighbors:
                lbp_features.append(np.mean(neighbor > center))
            
            feature_vector.extend(lbp_features)
            
            # Histogram features
            hist, _ = np.histogram(img_2d.flatten(), bins=8, range=(0, 1))
            hist = hist / np.sum(hist)  # Normalize
            feature_vector.extend(hist)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Complete data preparation pipeline."""
        # Load images and labels
        images, labels = self.load_images_and_labels()
        
        # Extract features
        features = self.extract_features(images)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocessor = ImageDataPreprocessor("iuss-23-24-automatic-diagnosis-breast-cancer")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")