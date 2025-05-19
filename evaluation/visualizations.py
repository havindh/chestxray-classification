"""
Visualization utilities for X-ray classification results.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import logging

logger = logging.getLogger(__name__)

def plot_results(results_df, test_df, target_labels, output_path=None):
    """
    Plot AUC results and dataset statistics
    
    Parameters:
    results_df (pandas.DataFrame): Results dataframe with AUC scores
    test_df (pandas.DataFrame): Test dataset
    target_labels (list): Names of target labels
    output_path (str, optional): Path to save visualizations
    """
    try:
        # Create a figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: AUC scores
        results_df.plot(x="Disease", y="ROC-AUC", kind="bar", ax=axes[0], color="skyblue", alpha=0.7)
        results_df.plot(x="Disease", y="PR-AUC", kind="bar", ax=axes[0], color="lightgreen", alpha=0.7)
        axes[0].set_title("AUC Scores by Disease")
        axes[0].set_ylabel("AUC")
        axes[0].set_ylim(0, 1)
        axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)  # Random baseline
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Dataset distribution
        test_df[target_labels].sum().plot(kind="bar", ax=axes[1], color="lightgreen")
        axes[1].set_title("Positive Case Count per Disease")
        axes[1].set_ylabel("Count")
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, "evaluation_results.png"))
        
        plt.show()
    
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        print(f"Couldn't create plots: {e}")

def plot_roc_curves(y_true, y_pred, target_labels, output_path=None):
    """
    Plot ROC curves for each disease
    
    Parameters:
    y_true (numpy.ndarray): Ground truth labels, shape [n_samples, n_classes]
    y_pred (numpy.ndarray): Predicted probabilities, shape [n_samples, n_classes]
    target_labels (list): Names of target labels
    output_path (str, optional): Path to save visualizations
    """
    try:
        plt.figure(figsize=(10, 8))
        
        for i, label in enumerate(target_labels):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            plt.plot(fpr, tpr, lw=2, label=f'{label}')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, "roc_curves.png"))
        
        plt.show()
    
    except Exception as e:
        logger.error(f"Error plotting ROC curves: {e}")
        print(f"Couldn't plot ROC curves: {e}")

def plot_pr_curves(y_true, y_pred, target_labels, output_path=None):
    """
    Plot Precision-Recall curves for each disease
    
    Parameters:
    y_true (numpy.ndarray): Ground truth labels, shape [n_samples, n_classes]
    y_pred (numpy.ndarray): Predicted probabilities, shape [n_samples, n_classes]
    target_labels (list): Names of target labels
    output_path (str, optional): Path to save visualizations
    """
    try:
        plt.figure(figsize=(10, 8))
        
        for i, label in enumerate(target_labels):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            plt.plot(recall, precision, lw=2, label=f'{label}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, "pr_curves.png"))
        
        plt.show()
    
    except Exception as e:
        logger.error(f"Error plotting PR curves: {e}")
        print(f"Couldn't plot PR curves: {e}")
