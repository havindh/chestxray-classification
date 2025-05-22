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
       # Load as PIL grayscale image
       img = Image.open(img_path).convert('L')
       
       # Convert to numpy array for xrv.datasets.normalize
       img = np.array(img)
       
       # Normalize using TorchXRayVision
       img = xrv.datasets.normalize(img, 255)
       
       # Convert to tensor and add channel dimension
       img = torch.from_numpy(img).float().unsqueeze(0)
       
       # Resize
       img = transforms.Resize((img_size, img_size))(img)
       
       return img
    except Exception as e:
       logger.error(f"Failed to preprocess image {img_path}: {e}")
       raise