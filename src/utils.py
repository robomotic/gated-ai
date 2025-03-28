import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def calculate_normalization_stats(dataset):
    """
    Calculate mean and standard deviation from a dataset.
    
    Args:
        dataset: PyTorch dataset (without normalization transform)
    
    Returns:
        mean, std: Calculated statistics
    """
    loader = DataLoader(dataset, batch_size=1000, num_workers=4, shuffle=False)
    mean = 0.
    std = 0.
    total_samples = 0
    
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean, std

def get_normalization_stats(dataset_name, dataset, recalculate=False):
    """
    Get normalization statistics for a dataset. If stats are already cached,
    they will be loaded from disk, otherwise they'll be calculated.
    
    Args:
        dataset_name: Name of the dataset (for caching)
        dataset: PyTorch dataset to calculate stats from (if needed)
        recalculate: Force recalculation even if cache exists
    
    Returns:
        mean, std: Normalization parameters
    """
    stats_dir = os.path.join('data', 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, f"normalization_stats_{dataset_name}.json")
    
    if not recalculate and os.path.exists(stats_file):
        print(f"Loading cached normalization stats for {dataset_name}...")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            return stats['mean'], stats['std']
    
    print(f"Calculating normalization stats for {dataset_name}...")
    mean, std = calculate_normalization_stats(dataset)
    
    # Save for future use
    with open(stats_file, 'w') as f:
        json.dump({'mean': mean.item(), 'std': std.item()}, f)
    
    print(f"Dataset {dataset_name} - mean: {mean.item():.4f}, std: {std.item():.4f}")
    return mean.item(), std.item()

def create_transforms(mean, std):
    """
    Create a transform pipeline with given normalization parameters.
    
    Args:
        mean: Mean value for normalization
        std: Standard deviation for normalization
    
    Returns:
        transform: Composed transform including ToTensor and Normalize
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
