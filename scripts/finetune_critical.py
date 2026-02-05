#!/usr/bin/env python3
"""
Fine-tune model ONLY on critical data (turns/S-shapes).

Usage:
    python scripts/finetune_critical.py --base_model steer_net.pth --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os
import sys
import argparse

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)
from steerDS import SteerDataSet

# ============================================================================
# CONFIGURATION
# ============================================================================
parser = argparse.ArgumentParser(description='Fine-tune on critical data only')
parser.add_argument('--base_model', type=str, default='steer_net.pth', help='Existing model to fine-tune')
parser.add_argument('--epochs', type=int, default=10, help='Fine-tuning epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (10x lower than baseline 0.01)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--output', type=str, default='steer_net_finetuned.pth', help='Output model path')
args = parser.parse_args()

# Critical data folders (recursive search will find all images)
CRITICAL_FOLDERS = [
    'best_data_trained_straight_model/green_anticlockwise',
    'best_data_trained_straight_model/green_clockwise',
    'best_data_trained_straight_model/orange_anticlockwise',
    'best_data_trained_straight_model/orange_clockwise',
]

print("="*60)
print("FINE-TUNING ON CRITICAL DATA ONLY")
print("="*60)
print(f"Base model: {args.base_model}")
print(f"Learning rate: {args.lr}")
print(f"Batch size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Output: {args.output}")
print("="*60)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model architecture (must match baseline)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load existing model
model_path = os.path.join(script_path, '..', args.base_model)
if not os.path.exists(model_path):
    model_path = args.base_model
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

net = Net().to(device)
net.load_state_dict(torch.load(model_path, map_location=device))
print(f"Loaded base model from: {model_path}")

# Transform (must match training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((40, 60)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ============================================================================
# LOAD CRITICAL DATA (recursive search in each folder)
# ============================================================================
print("\nLoading critical data folders...")
datasets = []
for folder in CRITICAL_FOLDERS:
    folder_path = os.path.join(script_path, '..', 'data', folder)
    if os.path.exists(folder_path):
        ds = SteerDataSet(folder_path, '.jpg', transform, recursive=True)
        datasets.append(ds)
        print(f"  {folder}: {len(ds)} images")
    else:
        print(f"  WARNING: {folder} not found, skipping")

if not datasets:
    print("ERROR: No critical data found!")
    sys.exit(1)

# Combine all critical datasets
critical_ds = ConcatDataset(datasets)
print(f"\nTotal critical data: {len(critical_ds)} images")

# DataLoader
trainloader = DataLoader(
    critical_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
print(f"Batches per epoch: {len(trainloader)}")

# ============================================================================
# FINE-TUNING (no class weighting - critical data is already turn-heavy)
# ============================================================================
criterion = nn.CrossEntropyLoss()  # No weights - uniform
optimizer = optim.Adam(net.parameters(), lr=args.lr)

print(f"\nStarting fine-tuning for {args.epochs} epochs...")
print("-"*60)

best_loss = float('inf')
for epoch in range(args.epochs):
    net.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = epoch_loss / len(trainloader)
    accuracy = 100 * correct / total
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        output_path = os.path.join(script_path, '..', args.output)
        torch.save(net.state_dict(), output_path)
        print(f"Epoch {epoch+1:2d}/{args.epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.1f}% [SAVED]")
    else:
        print(f"Epoch {epoch+1:2d}/{args.epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.1f}%")

print("-"*60)
print(f"\n✓ Fine-tuning complete!")
print(f"✓ Best model saved to: {args.output}")

# ============================================================================
# QUICK VALIDATION ON CRITICAL DATA
# ============================================================================
print("\n" + "="*60)
print("VALIDATION ON CRITICAL DATA")
print("="*60)

CLASS_LABELS = ["sharp_left", "left", "straight", "right", "sharp_right"]

net.eval()
class_correct = {label: 0 for label in CLASS_LABELS}
class_total = {label: 0 for label in CLASS_LABELS}

with torch.no_grad():
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        
        for label, pred in zip(labels, predicted):
            class_total[CLASS_LABELS[label.item()]] += 1
            if label == pred:
                class_correct[CLASS_LABELS[label.item()]] += 1

print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
print("-"*45)
for label in CLASS_LABELS:
    if class_total[label] > 0:
        acc = 100 * class_correct[label] / class_total[label]
        print(f"{label:<15} {class_correct[label]:<10} {class_total[label]:<10} {acc:.1f}%")

total_correct = sum(class_correct.values())
total_samples = sum(class_total.values())
print("-"*45)
print(f"{'OVERALL':<15} {total_correct:<10} {total_samples:<10} {100*total_correct/total_samples:.1f}%")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print(f"1. Update deploy.py to use '{args.output}'")
print(f"2. Test on the robot")
print(f"3. Run diagnose_model.py on FULL base data to check for forgetting:")
print(f"   python scripts/diagnose_model.py --model {args.output} --data best_data_trained_straight_model")
