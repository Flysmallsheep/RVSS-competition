#!/usr/bin/env python3
"""
Model Diagnostic Script
Run this after training to understand where your model is failing.

Usage:
    python scripts/diagnose_model.py --model steer_net.pth --data best_data_trained_straight_model
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os
import sys
import argparse
import cv2
from collections import defaultdict

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)
from steerDS import SteerDataSet

# Parse arguments
parser = argparse.ArgumentParser(description='Diagnose model performance')
parser.add_argument('--model', type=str, default='steer_net.pth', help='Path to model weights')
parser.add_argument('--data', type=str, default='best_data_trained_straight_model', help='Data folder name')
parser.add_argument('--save_misclassified', type=int, default=10, help='Number of misclassified images to save per class')
parser.add_argument('--output_dir', type=str, default='diagnostics', help='Output directory for diagnostic images')
args = parser.parse_args()

# Class labels
CLASS_LABELS = ["sharp_left", "left", "straight", "right", "sharp_right"]
CLASS_ANGLES = [-0.5, -0.25, 0.0, 0.25, 0.5]

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model architecture (must match training)
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

# Load model
model_path = os.path.join(script_path, '..', args.model)
if not os.path.exists(model_path):
    model_path = args.model  # Try as absolute path
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

net = Net().to(device)
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()
print(f"Loaded model from: {model_path}")

# Transform (must match training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((40, 60)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load dataset
data_path = os.path.join(script_path, '..', 'data', args.data)
dataset = SteerDataSet(data_path, '.jpg', transform)
print(f"Loaded {len(dataset)} images from {data_path}")

# Create output directory
output_dir = os.path.join(script_path, '..', args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# Run inference on all data
print("\nRunning inference on all data...")
all_true = []
all_pred = []
all_probs = []
all_files = dataset.filenames

misclassified = defaultdict(list)  # {(true_class, pred_class): [(file, probs), ...]}

with torch.no_grad():
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = img.unsqueeze(0).to(device)
        
        outputs = net(img)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        pred = np.argmax(probs)
        
        all_true.append(label)
        all_pred.append(pred)
        all_probs.append(probs)
        
        if pred != label:
            misclassified[(label, pred)].append((all_files[i], probs, probs[pred]))

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_probs = np.array(all_probs)

# ============================================================================
# DIAGNOSTIC 1: Overall Accuracy
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 1: Overall Accuracy")
print("="*60)
overall_acc = (all_true == all_pred).mean() * 100
print(f"Overall Accuracy: {overall_acc:.1f}%")

# ============================================================================
# DIAGNOSTIC 2: Per-Class Accuracy
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 2: Per-Class Accuracy")
print("="*60)
print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
print("-"*45)

class_accs = {}
for i, label in enumerate(CLASS_LABELS):
    mask = all_true == i
    total = mask.sum()
    correct = (all_pred[mask] == i).sum()
    acc = correct / total * 100 if total > 0 else 0
    class_accs[label] = acc
    
    status = "⚠️ LOW" if acc < 80 else "✓"
    print(f"{label:<15} {correct:<10} {total:<10} {acc:<10.1f}% {status}")

# ============================================================================
# DIAGNOSTIC 3: Confusion Matrix
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 3: Confusion Matrix")
print("="*60)

cm = metrics.confusion_matrix(all_true, all_pred, normalize='true')
print("\nNormalized confusion matrix (rows=true, cols=predicted):")
print(f"{'':>12}", end="")
for label in CLASS_LABELS:
    print(f"{label[:8]:>10}", end="")
print()

for i, label in enumerate(CLASS_LABELS):
    print(f"{label:<12}", end="")
    for j in range(len(CLASS_LABELS)):
        val = cm[i, j]
        # Highlight off-diagonal high values (confusion)
        marker = "**" if (i != j and val > 0.1) else "  "
        print(f"{val:>8.2f}{marker}", end="")
    print()

print("\n** = Significant confusion (>10%)")

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(10, 8))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
disp.plot(ax=ax, cmap='Blues', values_format='.2f')
plt.title('Confusion Matrix (Normalized)')
plt.tight_layout()
cm_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"\nSaved confusion matrix to: {cm_path}")

# ============================================================================
# DIAGNOSTIC 4: Confidence Analysis
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 4: Confidence Analysis")
print("="*60)

correct_mask = all_true == all_pred
wrong_mask = ~correct_mask

correct_conf = all_probs[correct_mask].max(axis=1).mean()
wrong_conf = all_probs[wrong_mask].max(axis=1).mean() if wrong_mask.sum() > 0 else 0

print(f"Avg confidence on CORRECT predictions: {correct_conf:.2f}")
print(f"Avg confidence on WRONG predictions:   {wrong_conf:.2f}")

if wrong_conf > 0.6:
    print("⚠️ Model is CONFIDENT but WRONG - likely label inconsistency or ambiguous data")
elif wrong_conf < 0.4:
    print("✓ Model is uncertain when wrong - might benefit from more diverse data")

# ============================================================================
# DIAGNOSTIC 5: Most Common Confusions
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 5: Most Common Confusions")
print("="*60)

confusion_counts = defaultdict(int)
for (true_cls, pred_cls), items in misclassified.items():
    confusion_counts[(CLASS_LABELS[true_cls], CLASS_LABELS[pred_cls])] = len(items)

sorted_confusions = sorted(confusion_counts.items(), key=lambda x: -x[1])[:10]
print(f"{'True Class':<15} {'Predicted':<15} {'Count':<10}")
print("-"*40)
for (true_cls, pred_cls), count in sorted_confusions:
    print(f"{true_cls:<15} {pred_cls:<15} {count:<10}")

# ============================================================================
# DIAGNOSTIC 6: Save Misclassified Examples
# ============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 6: Misclassified Examples")
print("="*60)

misclassified_dir = os.path.join(output_dir, 'misclassified')
os.makedirs(misclassified_dir, exist_ok=True)

# Save worst misclassified examples (sorted by confidence in wrong prediction)
saved_count = 0
for (true_cls, pred_cls), items in misclassified.items():
    # Sort by confidence in wrong prediction (descending)
    items_sorted = sorted(items, key=lambda x: -x[2])[:args.save_misclassified]
    
    true_label = CLASS_LABELS[true_cls]
    pred_label = CLASS_LABELS[pred_cls]
    
    for i, (filepath, probs, conf) in enumerate(items_sorted):
        # Read original image (before crop)
        try:
            img = cv2.imread(filepath)
            if img is None:
                continue
            
            # Add annotation
            img_annotated = img.copy()
            cv2.putText(img_annotated, f"TRUE: {true_label}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_annotated, f"PRED: {pred_label} ({conf:.2f})", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Save
            filename = f"{true_label}_as_{pred_label}_{i:02d}_conf{conf:.2f}.jpg"
            save_path = os.path.join(misclassified_dir, filename)
            cv2.imwrite(save_path, img_annotated)
            saved_count += 1
        except Exception as e:
            print(f"Error saving {filepath}: {e}")

print(f"Saved {saved_count} misclassified examples to: {misclassified_dir}")

# ============================================================================
# DIAGNOSTIC 7: Summary & Recommendations
# ============================================================================
print("\n" + "="*60)
print("SUMMARY & RECOMMENDATIONS")
print("="*60)

problems_found = []

# Check for low accuracy classes
low_acc_classes = [k for k, v in class_accs.items() if v < 80]
if low_acc_classes:
    problems_found.append(f"Low accuracy on: {', '.join(low_acc_classes)}")
    print(f"⚠️ Problem: Low accuracy on {low_acc_classes}")
    print(f"   → Collect more data for these classes, or check label consistency")

# Check for high confusion between adjacent classes
adjacent_confusion = []
for i in range(len(CLASS_LABELS) - 1):
    if cm[i, i+1] > 0.15 or cm[i+1, i] > 0.15:
        adjacent_confusion.append((CLASS_LABELS[i], CLASS_LABELS[i+1]))

if adjacent_confusion:
    problems_found.append(f"Confusion between adjacent classes")
    print(f"⚠️ Problem: Confusion between adjacent classes: {adjacent_confusion}")
    print(f"   → Consider merging classes (e.g., 5 → 3 classes) or collecting clearer examples")

# Check for high confidence on wrong predictions
if wrong_conf > 0.6:
    problems_found.append("High confidence on wrong predictions")
    print(f"⚠️ Problem: Model is confident when wrong (avg conf: {wrong_conf:.2f})")
    print(f"   → Likely cause: Label inconsistency in training data")
    print(f"   → Check the misclassified/ folder for examples")

# Check for class imbalance issues
max_acc = max(class_accs.values())
min_acc = min(class_accs.values())
if max_acc - min_acc > 20:
    problems_found.append("Large accuracy gap between classes")
    print(f"⚠️ Problem: Accuracy varies widely ({min_acc:.0f}% to {max_acc:.0f}%)")
    print(f"   → Class weights may not be strong enough, or need more data for weak classes")

if not problems_found:
    print("✓ No major problems detected! Model looks well-balanced.")
else:
    print(f"\nFound {len(problems_found)} potential issues to investigate.")

print(f"\n📁 Check the '{args.output_dir}/' folder for visualizations")
print("="*60)
