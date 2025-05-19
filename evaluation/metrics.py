"""
Evaluation metrics for X-ray classification.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging

logger = logging.getLogger(__name__)

def evaluate_predictions(y_true, y_pred, target_labels, output_path=None):
    """
    Calculate and report AUC scores with proper error handling
    
    Parameters:
    y_true (numpy.ndarray): Ground truth labels, shape [n_samples, n_classes]
    y_pred (numpy.ndarray): Predicted probabilities, shape [n_samples, n_classes]
    target_labels (list): Names of target labels
    output_path (str, optional): Path to save results
    
    Returns:
    pandas.DataFrame: Results dataframe with AUC scores
    """
    try:
        # Check shapes
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        logger.info(f"Evaluating on {len(y_true)} samples")
        
        # Calculate and report AUCs
        results = []
        print("\n=== Per-label AUC scores ===")
        
        for i, label in enumerate(target_labels):
            try:
                # ROC AUC
                roc_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                
                # PR AUC (Precision-Recall AUC)
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
                pr_auc = auc(recall, precision)
                
                print(f"{label}: ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")
                results.append((label, roc_auc, pr_auc))
            except ValueError as e:
                logger.error(f"Couldn't compute AUC for {label}: {e}")
                print(f"{label}: AUC could not be computed (check class distribution)")
        
        # Calculate mean AUC
        valid_roc_aucs = [roc_auc for _, roc_auc, _ in results]
        valid_pr_aucs = [pr_auc for _, _, pr_auc in results]
        
        if valid_roc_aucs:
            mean_roc_auc = np.mean(valid_roc_aucs)
            mean_pr_auc = np.mean(valid_pr_aucs)
            print(f"Mean ROC-AUC: {mean_roc_auc:.4f}, Mean PR-AUC: {mean_pr_auc:.4f}")
        
        # Save results to file
        results_df = pd.DataFrame(results, columns=["Disease", "ROC-AUC", "PR-AUC"])
        
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            results_df.to_csv(os.path.join(output_path, "auc_results.csv"), index=False)
        
        return results_df
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise