#!/usr/bin/env python3
"""
Quick script to check class distribution in your training data.
Run this to see if your data is imbalanced (common cause of "always turns one way").

Usage:
    pixi run python scripts/check_data_balance.py
"""

import os
from pathlib import Path
from collections import Counter

def get_steering_class(steering: float) -> int:
    """Same logic as steerDS.py"""
    if steering <= -0.5:
        return 0  # sharp left
    elif steering < 0:
        return 1  # left
    elif steering == 0:
        return 2  # straight
    elif steering < 0.5:
        return 3  # right
    else:
        return 4  # sharp right

def analyze_folder(folder_path: str):
    """Analyze class distribution in a folder of images."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return
    
    files = list(folder.rglob("*.jpg"))
    print(f"\nAnalyzing: {folder_path}")
    print(f"Total images: {len(files)}")
    
    if len(files) == 0:
        print("No .jpg files found!")
        return
    
    class_counts = Counter()
    steering_values = []
    parse_errors = 0
    
    for f in files:
        # Extract steering from filename (format: 000000X.XX.jpg or 000000-X.XX.jpg)
        name = f.stem  # filename without extension
        try:
            # steering is after the 6-digit image number
            steering_str = name[6:]  # e.g., "0.00" or "-0.50"
            steering = float(steering_str)
            steering_values.append(steering)
            cls = get_steering_class(steering)
            class_counts[cls] += 1
        except (ValueError, IndexError) as e:
            parse_errors += 1
            if parse_errors <= 5:
                print(f"  Parse error: {f.name} -> {e}")
    
    if parse_errors > 5:
        print(f"  ... and {parse_errors - 5} more parse errors")
    
    class_labels = ["sharp_left", "left", "straight", "right", "sharp_right"]
    
    print("\nClass distribution:")
    total = sum(class_counts.values())
    for i, label in enumerate(class_labels):
        count = class_counts.get(i, 0)
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {i} {label:12s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Show steering angle distribution
    if steering_values:
        import numpy as np
        arr = np.array(steering_values)
        print(f"\nSteering stats:")
        print(f"  min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, std={arr.std():.2f}")
        
        # Check for imbalance warning
        if total > 0:
            max_pct = max(class_counts.values()) / total * 100
            if max_pct > 50:
                print(f"\n⚠️  WARNING: Class imbalance detected! One class is {max_pct:.1f}% of data.")
                print("   This can cause the model to always predict that class.")
                print("   Collect more data for underrepresented classes.")

if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_root = os.path.join(script_path, "..", "data")
    
    # Analyze all subfolders
    data_path = Path(data_root)
    if data_path.exists():
        subfolders = [p for p in data_path.iterdir() if p.is_dir()]
        
        if subfolders:
            for folder in sorted(subfolders):
                analyze_folder(str(folder))
        else:
            analyze_folder(str(data_path))
        
        # Also analyze combined
        print("\n" + "="*60)
        analyze_folder(str(data_root))
    else:
        print(f"Data folder not found: {data_root}")
