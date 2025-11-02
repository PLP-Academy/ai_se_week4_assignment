"""
Model evaluation and metrics calculation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model, class_names=['Benign', 'Malignant']):
        self.model = model
        self.class_names = class_names
        
    def evaluate_model(self, X_test, y_test, X_train=None, y_train=None):
        """Comprehensive model evaluation."""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print("=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy Score: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Micro): {f1_micro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Training accuracy if provided
        if X_train is not None and y_train is not None:
            train_accuracy = self.model.score(X_train, y_train)
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Overfitting Check: {train_accuracy - accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names=None, top_n=15, save_path=None):
        """Plot feature importance for tree-based models."""
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print(f"\nTop {top_n} Most Important Features:")
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx]:20s}: {importances[idx]:.4f}")

def get_feature_names():
    """Get feature names for the extracted features."""
    base_features = [
        'Mean_Intensity', 'Std_Intensity', 'Var_Intensity',
        'Min_Intensity', 'Max_Intensity', 'Median_Intensity',
        'Percentile_10', 'Percentile_25', 'Percentile_75', 'Percentile_90',
        'Peak_to_Peak', 'Mean_Squares',
        'Edge_Strength_X', 'Edge_Strength_Y', 'Edge_Variation_X', 'Edge_Variation_Y',
        'Gradient_Magnitude_Mean', 'Gradient_Magnitude_Std', 'Max_Gradient', 'Gradient_95th'
    ]
    
    # LBP features
    lbp_features = [f'LBP_{i}' for i in range(8)]
    
    # Histogram features
    hist_features = [f'Hist_Bin_{i}' for i in range(8)]
    
    return base_features + lbp_features + hist_features

if __name__ == "__main__":
    # Example usage
    print("Model evaluation module ready for use in Jupyter notebook")