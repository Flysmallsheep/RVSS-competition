#!/usr/bin/env python3
"""
Compound Weighted Sampler

This module creates a WeightedRandomSampler that addresses TWO imbalances:
1. Class imbalance (straight is overrepresented)
2. Scenario imbalance (critical tiles are underrepresented)

The compound weight for each sample is:
    weight = critical_multiplier × class_weight

Where:
    - critical_multiplier = 3.0 if image is from critical folder, else 1.0
    - class_weight = inverse frequency of the class (rare classes get higher weight)

Usage:
    from compound_sampler import create_compound_sampler
    sampler = create_compound_sampler(dataset, critical_multiplier=3.0)
    dataloader = DataLoader(dataset, batch_size=256, sampler=sampler)
"""

import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import os


# ============================================================================
# STEP 1: Define which folders are "critical"
# ============================================================================
# These are the folders containing images from tiles where the robot fails.
# The sampler will give these images higher weight during training.

CRITICAL_FOLDERS = [
    'green_anticlockwise',
    'green_clockwise', 
    'orange_anticlockwise',
    'orange_clockwise',
    'failed_piles',
    'final_data',
]


def is_critical_path(filepath):
    """
    Check if an image filepath is from a critical folder.
    
    Args:
        filepath: Full path to the image file
        
    Returns:
        True if the image is from a critical folder, False otherwise
        
    Example:
        >>> is_critical_path('/data/best_data/green_anticlockwise/000001-0.50.jpg')
        True
        >>> is_critical_path('/data/best_data/data_feb3.1/000001-0.50.jpg')
        False
    """
    # Check if any critical folder name appears in the path
    for folder in CRITICAL_FOLDERS:
        if folder in filepath:
            return True
    return False


# ============================================================================
# STEP 2: Single-pass collection of labels and paths (OPTIMIZED)
# ============================================================================

def collect_labels_and_paths(dataset):
    """
    Collect all labels and filepaths in a SINGLE pass through the dataset.
    
    This is much faster than iterating twice (once for class counts, once for weights).
    With 14k images, this saves ~10-20 seconds of redundant loading.
    
    Args:
        dataset: Dataset with .filenames attribute and returns (image, label)
        
    Returns:
        Tuple of (labels_list, filepaths_list, class_counts)
    """
    labels = []
    filepaths = dataset.filenames  # Already available, no need to iterate
    class_counts = Counter()
    
    print("Collecting labels (single pass)...")
    for i in range(len(dataset)):
        _, label = dataset[i]
        # Handle both tensor and int labels
        if isinstance(label, torch.Tensor):
            label = label.item()
        labels.append(label)
        class_counts[label] += 1
        
        # Progress indicator for large datasets
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples...")
    
    print(f"Class counts: {dict(sorted(class_counts.items()))}")
    
    return labels, filepaths, class_counts


def calculate_class_weights_from_counts(class_counts):
    """
    Calculate inverse-frequency weights from pre-computed class counts.
    
    Args:
        class_counts: Counter or dict mapping class_id -> count
        
    Returns:
        Dictionary mapping class_id -> weight
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_id, count in class_counts.items():
        # Inverse frequency: rare classes get higher weight
        class_weights[class_id] = total_samples / (num_classes * count)
    
    print(f"Class weights (inverse frequency): {class_weights}")
    
    return class_weights


# ============================================================================
# STEP 3: Create compound weights (using cached data)
# ============================================================================

def create_compound_weights(labels, filepaths, class_weights, critical_multiplier=3.0):
    """
    Create compound weight for each sample using pre-collected labels and paths.
    
    Compound weight = critical_multiplier × class_weight
    
    This addresses both imbalances:
    - Class imbalance: rare classes weighted higher
    - Scenario imbalance: critical tiles weighted higher
    
    Args:
        labels: List of class labels for each sample
        filepaths: List of file paths for each sample
        class_weights: Dictionary mapping class_id -> weight
        critical_multiplier: How much more to weight critical samples (default 3.0)
        
    Returns:
        List of weights, one per sample
    """
    weights = []
    critical_count = 0
    non_critical_count = 0
    
    print(f"\nCreating compound weights (critical_multiplier={critical_multiplier})...")
    
    for i in range(len(labels)):
        filepath = filepaths[i]
        label = labels[i]
        
        # Factor 1: Critical multiplier (scenario emphasis)
        if is_critical_path(filepath):
            crit_weight = critical_multiplier
            critical_count += 1
        else:
            crit_weight = 1.0
            non_critical_count += 1
        
        # Factor 2: Class weight (class balance)
        cls_weight = class_weights[label]
        
        # Compound weight = product of both factors
        compound_weight = crit_weight * cls_weight
        weights.append(compound_weight)
    
    # Print summary statistics
    print(f"\nSample distribution:")
    print(f"  Critical samples: {critical_count} ({100*critical_count/len(labels):.1f}%)")
    print(f"  Non-critical samples: {non_critical_count} ({100*non_critical_count/len(labels):.1f}%)")
    
    print(f"\nWeight statistics:")
    print(f"  Min weight: {min(weights):.3f}")
    print(f"  Max weight: {max(weights):.3f}")
    print(f"  Mean weight: {sum(weights)/len(weights):.3f}")
    
    return weights


# ============================================================================
# STEP 4: Create the WeightedRandomSampler (MAIN FUNCTION)
# ============================================================================

def create_compound_sampler(dataset, critical_multiplier=3.0):
    """
    Create a WeightedRandomSampler with compound weights.
    
    OPTIMIZED: Uses single pass through dataset to collect labels,
    then computes all weights without re-iterating.
    
    Args:
        dataset: Dataset with .filenames attribute and returns (image, label)
        critical_multiplier: How much more to weight critical samples (default 3.0)
        
    Returns:
        WeightedRandomSampler that can be passed to DataLoader
        
    Usage:
        sampler = create_compound_sampler(train_dataset, critical_multiplier=3.0)
        train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
    """
    print("="*60)
    print("CREATING COMPOUND WEIGHTED SAMPLER")
    print("="*60)
    
    # Step 1: Single pass to collect all labels and paths
    labels, filepaths, class_counts = collect_labels_and_paths(dataset)
    
    # Step 2: Calculate class weights from counts
    class_weights = calculate_class_weights_from_counts(class_counts)
    
    # Step 3: Create compound weights (no dataset iteration needed)
    weights = create_compound_weights(labels, filepaths, class_weights, critical_multiplier)
    
    # Step 4: Create the sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    print(f"\n✓ Sampler created with {len(weights)} weights")
    print("="*60)
    
    return sampler


# ============================================================================
# STEP 5: Utility to verify the sampler is working
# ============================================================================

def verify_sampler(dataset, sampler, num_batches=10, batch_size=256):
    """
    Verify that the sampler produces the expected distribution.
    
    Samples a few batches and prints the actual distribution of:
    - Critical vs non-critical samples
    - Class distribution
    
    This helps confirm the compound weighting is working correctly.
    
    Args:
        dataset: The dataset being sampled
        sampler: The WeightedRandomSampler to verify
        num_batches: How many batches to sample for verification
        batch_size: Batch size to simulate
    """
    from torch.utils.data import DataLoader
    
    print("\n" + "="*60)
    print("VERIFYING SAMPLER DISTRIBUTION")
    print("="*60)
    
    # Create a dataloader with the sampler
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # Count distributions
    critical_count = 0
    non_critical_count = 0
    class_counts = Counter()
    
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
            
        for i, label in enumerate(labels):
            label = label.item() if isinstance(label, torch.Tensor) else label
            class_counts[label] += 1
            total_samples += 1
            
            # We can't easily get filepath from batch, so skip critical count here
    
    print(f"\nSampled {total_samples} images across {num_batches} batches:")
    print(f"\nClass distribution in sampled batches:")
    class_labels = ["sharp_left", "left", "straight", "right", "sharp_right"]
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        pct = 100 * count / total_samples
        label = class_labels[class_id] if class_id < len(class_labels) else f"class_{class_id}"
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)


# ============================================================================
# Example usage (when run directly)
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from steerDS import SteerDataSet
    import torchvision.transforms as transforms
    
    # Setup transform (must match training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((40, 60)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load dataset
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, '..', 'data', 'best_data_trained_straight_model')
    
    print(f"Loading dataset from: {data_path}")
    dataset = SteerDataSet(data_path, '.jpg', transform, recursive=True)
    print(f"Loaded {len(dataset)} images")
    
    # Create compound sampler
    sampler = create_compound_sampler(dataset, critical_multiplier=3.0)
    
    # Verify it's working
    verify_sampler(dataset, sampler, num_batches=10, batch_size=256)
