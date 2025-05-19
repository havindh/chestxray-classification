"""
Model loading utilities for X-ray classification.
"""

import torch
import torchxrayvision as xrv
import logging

logger = logging.getLogger(__name__)

def load_models(device):
    """
    Load pretrained X-ray models with proper error handling
    
    Parameters:
    device (torch.device): Device to load models on
    
    Returns:
    tuple: (model_mimic, model_chex, pathologies)
    """
    try:
        logger.info("Loading DenseNet121 pretrained on MIMIC-CXR...")
        model_mimic = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch").eval().to(device)
        
        logger.info("Loading DenseNet121 pretrained on CheXpert...")
        model_chex = xrv.models.DenseNet(weights="densenet121-res224-chex").eval().to(device)
        
        # Confirm label consistency between models
        if model_mimic.pathologies != model_chex.pathologies:
            raise ValueError("Label mismatch between pretrained models!")
        
        logger.info(f"Models loaded successfully with {len(model_mimic.pathologies)} disease classes")
        return model_mimic, model_chex, model_mimic.pathologies
    
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def list_available_models():
    """
    List all available pretrained models in TorchXRayVision
    
    Returns:
    dict: Available model weights
    """
    try:
        model_urls = xrv.models.model_urls
        return {
            "name": list(model_urls.keys()),
            "urls": list(model_urls.values())
        }
    except Exception as e:
        logger.error(f"Failed to list available models: {e}")
        return {}     