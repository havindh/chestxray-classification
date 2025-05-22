"""
Image preprocessing utilities for X-ray images.
"""

import torch
import torchxrayvision as xrv
from torchvision import transforms
from PIL import Image
import logging
import numpy as np

logger = logging.getLogger(__name__)

def preprocess_xray(img_path, img_size=224):
    """
    Properly preprocess X-ray image for TorchXRayVision models
    
    Parameters:
    img_path (str): Path to the X-ray image
    img_size (int): Size to resize image to (square)
    
    Returns:
    torch.Tensor: Preprocessed image tensor of shape [1, img_size, img_size]
    """
    try:
        # Load as grayscale 
        # This is a PIL image we  should convert it to numpy arrrays
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        # Use TorchXRayVision's normalization function
        # This handles the conversion to the expected [-1024, 1024] range
        img = xrv.datasets.normalize(img, 255)
        
        # Resize to expected dimensions
        img = transforms.Resize((img_size, img_size))(img)
        
        # Convert to tensor - shape will be [1, img_size, img_size]
        # When applied after, it doesn't rescale to [0,1] 
        # because it already receives a normalized tensor, not a PIL Image with 0-255 values
        img = transforms.ToTensor()(img)
        
        return img
    except Exception as e:
        logger.error(f"Failed to preprocess image {img_path}: {e}")
        raise
