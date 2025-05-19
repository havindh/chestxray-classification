"""
Configuration settings for the X-ray classification project.
"""

import os
import torch

class Config:
    """Centralized configuration settings"""
    # Paths
    data_path = "/content/drive/Shareddrives/CS231N/assignment4/cs231n/datasets/nih-chestxray"
    image_path = os.path.join(data_path, "images")
    csv_path = os.path.join(data_path, "Data_Entry_2017_v2020.csv")
    output_path = "results"
    
    # Model settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_labels = ["Cardiomegaly", "Atelectasis", "Effusion", "Pneumothorax"]
    img_size = 224
    batch_size = 16  # Added batching for efficiency: processes multiple images at once for efficiency
    
    # Dataset settings
    samples_per_disease = 400 # balances the dataset with 400 examples per class
    random_seed = 42 #ensures reproducible results
    
    # For Colab environment
    @classmethod
    def setup_colab_paths(cls, drive_path):
        """Update paths for Google Colab environment"""
        cls.data_path = os.path.join(drive_path, "CS231N/assignment4/cs231n/datasets/nih-chestxray")
        cls.image_path = os.path.join(cls.data_path, "images")
        cls.csv_path = os.path.join(cls.data_path, "Data_Entry_2017_v2020.csv")