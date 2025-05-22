"""
Image preprocessing utilities for X-ray images.
"""

import torchxrayvision as xrv
import skimage, torch, torchvision
import logging

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
       # Prepare the image:
        img = skimage.io.imread(img_path)
        # convert 8-bit image to [-1024, 1024] range
        img = xrv.datasets.normalize(img, 255) 
        # img = img.mean(2)[None, ...] # Make single color channel
        # Make single color channel - handle both RGB and grayscale
        if len(img.shape) == 3:
            img = img.mean(2)[None, ...]  # RGB to grayscale: (H, W, 3) -> (1, H, W)
        else:
            img = img[None, ...]  # Already grayscale: (H, W) -> (1, H, W)

        # Apply TorchXRayVision transforms
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        
        img = transform(img)
        img = torch.from_numpy(img)
        
        # Add batch dimension: (1, H, W) -> (1, 1, H, W)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        return img
    except Exception as e:
       logger.error(f"Failed to preprocess image {img_path}: {e}")
       raise