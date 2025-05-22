"""
Data loading and preparation utilities for X-ray classification.
"""

import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def encode_labels(row, all_labels):
    """
    Encode multi-label findings as binary vector
    
    Parameters:
    row (pandas.Series): DataFrame row containing 'Finding Labels'
    all_labels (list): List of all possible labels
    
    Returns:
    list: Binary encoding for each label
    """
    findings = row["Finding Labels"].split("|")
    return [int(label in findings) for label in all_labels]

def load_and_prepare_data(config, pathologies):
    """
    Load and prepare dataset with proper error handling
    
    Parameters:
    config (Config): Configuration object with paths and settings
    pathologies (list): List of all pathologies supported by models
    
    Returns:
    tuple: (test_df, label_indices)
    """
    try:
        logger.info(f"Loading metadata from {config.csv_path}")
        metadata_df = pd.read_csv(config.csv_path)
        logger.info(f"Raw metadata contains {len(metadata_df)} entries")
        
        # Get available images
        available_images = {f for f in os.listdir(config.image_path) if f.endswith(".png")}
        logger.info(f"Found {len(available_images)} PNG images in directory")
        
        # Filter to match available images
        metadata_df = metadata_df[metadata_df["Image Index"].isin(available_images)]
        logger.info(f"Filtered to {len(metadata_df)} entries with available images")
        
        
        # First encode the labels using your encode_labels function
        metadata_df["encoded_labels"] = metadata_df.apply(
            lambda row: encode_labels(row, pathologies), axis=1
        )
        # Then convert to numpy array
        metadata_df["encoded_array"] = metadata_df["encoded_labels"].apply(np.array)
        
        
        # Create label_indices as a dictionary
        label_indices = [pathologies.index(label) for label in config.target_labels]
        
        # Helper to check if all target diseases are negative
        def is_all_target_negative(row):
            return sum(row[label_indices]) == 0

        metadata_df["is_all_target_negative"] = metadata_df["encoded_array"].apply(
            lambda row: is_all_target_negative(row)
        )
        # Create separate columns using the list with target labels
        for disease, idx in zip(config.target_labels, label_indices):
          metadata_df[disease] = metadata_df["encoded_array"].apply(lambda x: x[idx])
        metadata_df.to_csv(os.path.join(config.output_path, "nih_metadata.csv"), index=False)
        return metadata_df, label_indices
    
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        raise
def get_test_set(metadata_df, label_indices, config):
  # Create test set with balanced classes
  test_df = create_balanced_test_set(metadata_df, label_indices, config)
  return test_df

def create_balanced_test_set(metadata_df, label_indices, config):
    """
    Create a balanced test set with same number of samples per disease
    
    Parameters:
    metadata_df (pandas.DataFrame): DataFrame with encoded labels
    label_indices (list): Indices of target labels
    config (Config): Configuration object with settings
    
    Returns:
    pandas.DataFrame: Balanced test set
    """
    # Helper to check if a disease is positive
    def is_positive(label_index):
        return metadata_df["encoded_array"].apply(lambda x: x[label_index] == 1)
    
    # Sample positive examples for each disease
    # Sample 400 positive per disease
    positive_samples = []

    for idx, label in zip(label_indices, config.target_labels):
        positives = metadata_df[is_positive(idx)]
        # Handle case where there might be fewer than 400 samples
        n_samples = min(400, len(positives))
        sampled = positives.sample(n=n_samples, random_state=42)
        positive_samples.append(sampled)

    # Combine all disease-positive samples
    positive_samples = pd.concat(positive_samples)
    
    # Sample negative examples (zero for all target diseases)
    negatives = metadata_df[metadata_df["is_all_target_negative"]]
    logger.info(f"Found {len(negatives)} cases negative for all target diseases")
    if len(negatives) < config.samples_per_disease:
        logger.warning(
            f"Only {len(negatives)} negative samples available, "
            f"wanted {config.samples_per_disease}"
        )
        negative_df = negatives
    else:
        negative_df = negatives.sample(n=config.samples_per_disease, random_state=config.random_seed)
    
    # Combine, deduplicate and prepare test set
    test_df = pd.concat(positive_samples + [negative_df])
    test_df = test_df.drop_duplicates(subset="Image Index").reset_index(drop=True)
    logger.info(f"Created test set with {len(test_df)} unique samples after deduplication")
    
    # Extract individual columns for each target disease
    for i, label in zip(label_indices, config.target_labels):
        test_df[label] = test_df["encoded_array"].apply(lambda x: x[i])
    
    # Save test set for reproducibility
    os.makedirs(config.output_path, exist_ok=True)
    test_df[["Image Index"]].to_csv(os.path.join(config.output_path, "nih_test_set_balanced.csv"), index=False)
    
    return test_df