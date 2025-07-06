'''
Custom transform definitions for the MRNet dataset (multi-view, 3D knee MRIs)
'''


import numpy as np
import torch
import torchvision.transforms as T


class RandomIntensityScale:
    """
    RandomIntensityScale
    - Applies a random scaling factor to the entire volume intensity.
    - This simulates variations in scanner brightness and contrast.
    """

    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, volume):
        # Sample a random scale factor from the given range
        scale = np.random.uniform(*self.scale_range)
        return volume * scale


class AddGaussianNoise:
    """
    AddGaussianNoise
    - Adds Gaussian noise to the entire volume.
    - This simulates scanner noise and improves model robustness.
    """

    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, volume):
        # Generate noise tensor with the same shape as input volume
        noise = torch.randn_like(volume) * self.std + self.mean
        return volume + noise


class PadOrCrop:
    """
    A custom PyTorch transform to standardise the number of slices (depth).

    This transform ensures that each volume has exactly `target_slices`:
    - If the volume has fewer slices than the target, it pads with zeros at the end.
    - If the volume has more slices, it performs a central crop.
    - If the volume already has the target number of slices, it returns the volume unchanged.
    """
    def __init__(self, target_slices=32):
        self.target_slices = target_slices

    def __call__(self, volume_tensor):
        current_slices = volume_tensor.shape[1]
        if current_slices == self.target_slices: # Case 1: Already matches the target number of slices
            return volume_tensor
        elif current_slices < self.target_slices: # Case 2: Volume has fewer slices, pad with zeros at the end
            pad_slices = self.target_slices - current_slices
            padding = torch.zeros((volume_tensor.shape[0], pad_slices, volume_tensor.shape[2], volume_tensor.shape[3]))
            return torch.cat([volume_tensor, padding], dim=1)
        else: # Case 3: Volume has more slices, crop centrally
            start = (current_slices - self.target_slices) // 2
            return volume_tensor[:, start:start + self.target_slices]


# Training pipeline (augmentation + standardisation)
def get_mrnet_train_transforms(target_slices=32):
    """ 
    Returns a composed set of volume-level transforms for training.
    
    Transformations applied:
    - RandomHorizontalFlip: Introduces left-right anatomical variability, which can help generalise across patient anatomy.
    - RandomIntensityScale: Simulates scanner-dependent intensity variations (brightness/contrast differences).
    - AddGaussianNoise: Adds synthetic noise to mimic MRI acquisition variability.
    - PadOrCrop: Standardises the number of slices per volume to the fixed target_slices.

    Arguments:
        target_slices (int): Number of slices to pad or crop each MRI volume to.

    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for use in MRNet training.
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),                     # Random left-right flipping with 50% probability (left/right)
        RandomIntensityScale(scale_range=(0.9, 1.1)),      # Random intensity scaling to simulate contrast variation
        AddGaussianNoise(mean=0.0, std=0.02),              # Add Gaussian noise to mimic imaging noise
        PadOrCrop(target_slices)                           # Ensure all volumes have the same number of slices
    ])


# Validation pipeline (no augmentation, only shape standardisation)
def get_mrnet_valid_transforms(target_slices=32):
    """
    Returns a composed set of transforms for validation.
    
    - No data augmentation is applied here.
    - Only standardisation to ensure consistent input dimensions across the validation set.

    Arguments:
        target_slices (int): Number of slices to pad or crop each MRI volume to.

    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for use in MRNet validation.
    """
    return T.Compose([
        PadOrCrop(target_slices)
    ])