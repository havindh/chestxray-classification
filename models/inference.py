"""
Inference utilities for X-ray classification.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import logging

from utils.preprocessing import preprocess_xray


logger = logging.getLogger(__name__)

def process_batch(image_batch, model_mimic, model_chex, device):
    """
    Process a batch of images through both models and ensemble predictions
    
    Parameters:
    image_batch (list): List of image tensors
    model_mimic (torch.nn.Module): MIMIC-CXR pretrained model
    model_chex (torch.nn.Module): CheXpert pretrained model
    device (torch.device): Device to run inference on
    
    Returns:
    numpy.ndarray: Ensemble predictions
    """
    with torch.no_grad():
        batch_tensor = torch.stack(image_batch).to(device)
        
        # Run both models
        out_mimic = model_mimic(batch_tensor)
        out_chex = model_chex(batch_tensor)
        
        # Average predictions (ensemble)
        avg_pred = (out_mimic + out_chex) / 2
        return avg_pred.cpu().numpy()  # Return as numpy array

def run_inference(test_df, model_mimic, model_chex, label_indices, config):
    """
    Run inference with batching for improved efficiency
    
    Parameters:
    test_df (pandas.DataFrame): Test dataset
    model_mimic (torch.nn.Module): MIMIC-CXR pretrained model
    model_chex (torch.nn.Module): CheXpert pretrained model
    label_indices (list): Indices of target labels
    config (Config): Configuration object
    
    Returns:
    tuple: (y_true, y_pred) numpy arrays
    """
    current_batch = []
    batch_labels = []
    y_true = []
    y_pred = []
    
    skipped_images = 0
    
    logger.info(f"Running inference on {len(test_df)} images")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        img_path = os.path.join(config.image_path, row["Image Index"])
        
        try:
            # Preprocess image
            img_tensor = preprocess_xray(img_path, config.img_size)
            current_batch.append(img_tensor)
            batch_labels.append(np.array(row["encoded_array"])[label_indices])
            
            # Process batch when it reaches the specified batch size
            if len(current_batch) == config.batch_size:
                # Process the batch
                batch_preds = process_batch(current_batch, model_mimic, model_chex, config.device)
                
                # Extract predictions for target labels
                for i, pred in enumerate(batch_preds):
                    y_pred.append(pred[label_indices])
                    y_true.append(batch_labels[i])
                
                # Reset batch
                current_batch = []
                batch_labels = []
        
        except Exception as e:
            logger.warning(f"Skipping {row['Image Index']}: {e}")
            skipped_images += 1
    
    # Process any remaining images
    if current_batch:
        batch_preds = process_batch(current_batch, model_mimic, model_chex, config.device)
        for i, pred in enumerate(batch_preds):
            y_pred.append(pred[label_indices])
            y_true.append(batch_labels[i])
    
    if skipped_images > 0:
        logger.warning(f"Skipped {skipped_images} images due to errors")
    
    return np.array(y_true), np.array(y_pred)