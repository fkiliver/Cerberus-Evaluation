import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelEvaluator:
    """A tool for evaluating model performance in the Cerberus framework."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("ModelEvaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("model_evaluation.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def evaluate_binary_classification(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate binary classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    def evaluate_multi_class(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None,
                           labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate multi-class classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            labels: List of class labels (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro')
        }
        
        if y_prob is not None:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
        
        # Generate detailed classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True
        )
        
        return metrics
    
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            title: str = "Confusion Matrix") -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of class labels (optional)
            title: Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save plot
        output_path = self.output_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved confusion matrix plot to: {output_path}")
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_prob: np.ndarray,
                      labels: Optional[List[str]] = None) -> None:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            labels: List of class labels (optional)
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # Binarize the output
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # Compute ROC curve and ROC area for each class
        plt.figure(figsize=(10, 8))
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            label = labels[i] if labels else f'Class {i}'
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        
        # Save plot
        output_path = self.output_dir / f"roc_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved ROC curves plot to: {output_path}")
    
    def generate_evaluation_report(self,
                                 metrics: Dict[str, Any],
                                 model_name: str,
                                 dataset_name: str,
                                 evaluation_time: datetime) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of evaluation metrics
            model_name: Name of the model
            dataset_name: Name of the dataset
            evaluation_time: Time of evaluation
            
        Returns:
            Markdown formatted report
        """
        report = f"""# Model Evaluation Report

## Overview
- Model: {model_name}
- Dataset: {dataset_name}
- Evaluation Time: {evaluation_time.strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics
"""
        
        # Add basic metrics
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                report += f"- {metric}: {value:.4f}\n"
        
        # Add classification report if available
        if 'classification_report' in metrics:
            report += "\n## Detailed Classification Report\n"
            report += "```\n"
            report += classification_report(
                metrics['classification_report'],
                target_names=metrics.get('labels', None)
            )
            report += "```\n"
        
        return report
    
    def save_evaluation_results(self,
                              metrics: Dict[str, Any],
                              model_name: str,
                              dataset_name: str) -> None:
        """
        Save evaluation results to files.
        
        Args:
            metrics: Dictionary of evaluation metrics
            model_name: Name of the model
            dataset_name: Name of the dataset
        """
        # Save metrics to JSON
        metrics_path = self.output_dir / f"{model_name}_{dataset_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate and save report
        report = self.generate_evaluation_report(
            metrics,
            model_name,
            dataset_name,
            datetime.now()
        )
        report_path = self.output_dir / f"{model_name}_{dataset_name}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Saved evaluation results to: {metrics_path}")
        self.logger.info(f"Saved evaluation report to: {report_path}")

def main():
    """Example usage of the ModelEvaluator."""
    evaluator = ModelEvaluator()
    
    # Example binary classification evaluation
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.3, 0.2, 0.9, 0.4, 0.7, 0.8])
    
    # Evaluate binary classification
    metrics = evaluator.evaluate_binary_classification(y_true, y_pred, y_prob)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(y_true, y_pred, labels=['Benign', 'Malicious'])
    
    # Save results
    evaluator.save_evaluation_results(metrics, "binary_classifier", "test_dataset")
    
    # Example multi-class evaluation
    y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred_multi = np.array([0, 1, 1, 0, 2, 2, 0, 1])
    y_prob_multi = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.7, 0.2],
        [0.9, 0.05, 0.05],
        [0.1, 0.2, 0.7],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1]
    ])
    
    # Evaluate multi-class classification
    metrics_multi = evaluator.evaluate_multi_class(
        y_true_multi,
        y_pred_multi,
        y_prob_multi,
        labels=['Class A', 'Class B', 'Class C']
    )
    
    # Plot confusion matrix and ROC curves
    evaluator.plot_confusion_matrix(
        y_true_multi,
        y_pred_multi,
        labels=['Class A', 'Class B', 'Class C']
    )
    evaluator.plot_roc_curve(y_true_multi, y_prob_multi, labels=['Class A', 'Class B', 'Class C'])
    
    # Save results
    evaluator.save_evaluation_results(metrics_multi, "multi_classifier", "test_dataset")

if __name__ == "__main__":
    main() 