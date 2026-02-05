import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import os
import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

from steerDS import SteerDataSet
from compound_sampler import create_compound_sampler  # Compound weighting: class balance + critical emphasis

#######################################################################################################################################
####     This tutorial is adapted from the PyTorch "Train a Classifier" tutorial                                                   ####
####     Please review here if you get stuck: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html                   ####
#######################################################################################################################################
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Helper function for visualising images in our dataset
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:,:,::-1]
    plt.imshow(rgbimg)
    plt.show()

#######################################################################################################################################
####     SETTING UP THE DATASET                                                                                                    ####
#######################################################################################################################################

#transformations for raw images before going to CNN
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((40, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

###################
## Train dataset ##
###################

from pathlib import Path

# OLD: make_weighted_sampler (class-only weighting) - REMOVED
# NEW: Using compound_sampler which handles BOTH:
#   1. Class imbalance (inverse frequency weighting)
#   2. Scenario imbalance (critical tiles get higher weight)
# See compound_sampler.py for implementation details.


def gather_all_images(root_dir: str, img_ext: str = ".jpg"):
    root = Path(root_dir)
    files = [str(p) for p in root.rglob(f"*{img_ext}")]
    return sorted(files)

def split_train_val_by_folder(files, train_ratio=0.9, seed=0):
    """
    Prevent leakage: all images from the same folder go to the same split.
    """
    rng = np.random.default_rng(seed)

    # group by parent folder
    parents = sorted(set(Path(f).parent for f in files))
    rng.shuffle(parents)

    n_train_folders = max(1, int(len(parents) * train_ratio))
    train_parents = set(parents[:n_train_folders])

    train_files = [f for f in files if Path(f).parent in train_parents]
    val_files   = [f for f in files if Path(f).parent not in train_parents]
    return train_files, val_files

#######################################################################################################################################
####     HYPERPARAMETERS - Tune these for your dataset (~14k images)                                                               ####
#######################################################################################################################################
BATCH_SIZE = 256          # Good balance: ~55 batches/epoch for 14k images
NUM_EPOCHS = 45           # Enough iterations to converge
LEARNING_RATE = 0.001     # Adam default, works well
WEIGHT_DECAY = 0          # L2 regularization to prevent overfitting (e.g., 1e-4)
CRITICAL_MULTIPLIER = 3.0 # How much more to weight critical tiles (turns/S-shapes)

# Early stopping parameters
EARLY_STOP_PATIENCE = 8   # Stop if val loss doesn't improve for this many epochs
EARLY_STOP_MIN_DELTA = 0.001  # Minimum improvement to count as "better"

script_path = os.path.dirname(os.path.realpath(__file__))

# CHANGE THIS: point to the folder that contains ALL subfolders of images
data_root = os.path.join(script_path, '..', 'data')  # <-- adjust if needed

all_files = gather_all_images(data_root, img_ext=".jpg")
print("Total images found:", len(all_files))

train_files, val_files = split_train_val_by_folder(all_files, train_ratio=0.9, seed=0)
print("Train images:", len(train_files))
print("Val images:", len(val_files))

train_ds = SteerDataSet(filenames=train_files, img_ext=".jpg", transform=transform)
val_ds   = SteerDataSet(filenames=val_files,   img_ext=".jpg", transform=transform)

# Create compound sampler: addresses BOTH class imbalance AND critical tile emphasis
train_sampler = create_compound_sampler(train_ds, critical_multiplier=CRITICAL_MULTIPLIER)

trainloader = DataLoader(
    train_ds,
    batch_size= BATCH_SIZE,
    sampler=train_sampler,  # <-- oversampling
    shuffle=False,          # <-- must be False when using sampler
)
valloader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


num_classes = 5
counts = torch.zeros(num_classes, dtype=torch.long)

with torch.no_grad():
    for _, y in valloader:
        y = y.view(-1).long().cpu()  # (B,) on CPU
        counts += torch.bincount(y, minlength=num_classes)

print("Val counts per class:", counts.tolist())
print("Class names:", val_ds.class_labels)


#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################

###########################################################
##########  Original CNN Model                   ##########
###########################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 5)

        self.relu = nn.ReLU()


    def forward(self, x):
        #extract features with convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        #linear layer for classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
       
        return x

###########################################################
##########  MobileNet V3 Small Model             ##########
###########################################################
# class Net(nn.Module):
#     def __init__(self, num_classes=5, pretrained=False, dropout=0.2, freeze_backbone=False):
#         super().__init__()
#         weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
#         self.model = mobilenet_v3_small(weights=weights)

#         in_features = self.model.classifier[-1].in_features
#         self.model.classifier[-1] = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(in_features, num_classes),
#         )

#         if freeze_backbone:
#             for p in self.model.features.parameters():
#                 p.requires_grad = False

#     def forward(self, x):
#         return self.model(x)



net = Net().to(device)

#######################################################################################################################################
####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
#######################################################################################################################################

#for classification tasks
criterion = nn.CrossEntropyLoss()
#You could use also ADAM
# optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ReduceLROnPlateau: reduces LR when validation loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # minimize validation loss
    factor=0.5,       # halve the LR when triggered
    patience=3,       # wait 3 epochs before reducing
    min_lr=1e-6       # don't go below this LR
)


#######################################################################################################################################
####     TRAINING LOOP                                                                                                             ####
#######################################################################################################################################

# Print training configuration summary
print("\n" + "="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"  Batch size:          {BATCH_SIZE}")
print(f"  Epochs:              {NUM_EPOCHS}")
print(f"  Learning rate:       {LEARNING_RATE}")
print(f"  Weight decay:        {WEIGHT_DECAY}")
print(f"  Critical multiplier: {CRITICAL_MULTIPLIER}")
print(f"  Early stop patience: {EARLY_STOP_PATIENCE}")
print(f"  Training samples:    {len(train_ds)}")
print(f"  Validation samples:  {len(val_ds)}")
print(f"  Batches per epoch:   {len(trainloader)}")
print(f"  Device:              {device}")
print("="*70 + "\n")

losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}
best_acc = 0
best_val_loss = float('inf')
epochs_without_improvement = 0
CLASS_LABELS = ["sharp_left", "left", "straight", "right", "sharp_right"]
for epoch in range(NUM_EPOCHS):
    # ================================================================
    # TRAINING PHASE
    # ================================================================
    net.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(trainloader)
    
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress indicator every 10 batches
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"\r  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {i+1}/{num_batches} | Loss: {loss.item():.4f}", end="")
    
    print()  # New line after progress
    
    train_loss = epoch_loss / len(trainloader)
    train_acc = 100. * correct / total
    losses['train'].append(train_loss)
    accs['train'].append(train_acc)

    # ================================================================
    # VALIDATION PHASE
    # ================================================================
    net.eval()
    correct_pred = {label: 0 for label in CLASS_LABELS}
    total_pred = {label: 0 for label in CLASS_LABELS}
    val_loss = 0
    
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            for label, prediction in zip(labels, predictions):
                label_name = CLASS_LABELS[label.item()]
                total_pred[label_name] += 1
                if label == prediction:
                    correct_pred[label_name] += 1

    # Calculate per-class accuracy
    class_accs = []
    for classname in CLASS_LABELS:
        if total_pred[classname] > 0:
            acc = 100 * correct_pred[classname] / total_pred[classname]
            class_accs.append(acc)
        else:
            class_accs.append(0.0)

    val_loss_avg = val_loss / len(valloader)
    val_acc_avg = np.mean(class_accs)
    losses['val'].append(val_loss_avg)
    accs['val'].append(val_acc_avg)

    # ================================================================
    # PRINT EPOCH SUMMARY WITH PER-CLASS ACCURACY
    # ================================================================
    print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
    print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
    print(f"    Val Loss:   {val_loss_avg:.4f} | Val Acc:   {val_acc_avg:.1f}%")
    print(f"    Per-class accuracy:")
    for i, classname in enumerate(CLASS_LABELS):
        status = "+" if class_accs[i] >= 90 else "~" if class_accs[i] >= 70 else "-"
        print(f"      {classname:<12}: {class_accs[i]:5.1f}% [{status}]")
    
    # Step scheduler with validation loss
    scheduler.step(val_loss_avg)
    print(f"    Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ================================================================
    # SAVE BEST MODEL
    # ================================================================
    if val_acc_avg > best_acc:
        torch.save(net.state_dict(), 'steer_net.pth')
        best_acc = val_acc_avg
        print(f"    >>> New best model saved! (Val Acc: {best_acc:.1f}%)")

    # ================================================================
    # EARLY STOPPING CHECK
    # ================================================================
    if val_loss_avg < best_val_loss - EARLY_STOP_MIN_DELTA:
        best_val_loss = val_loss_avg
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"    ! No improvement for {epochs_without_improvement}/{EARLY_STOP_PATIENCE} epochs")
    
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\n  EARLY STOPPING triggered after {epoch+1} epochs!")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        break
    
    print("-" * 60)

print('\n' + "="*70)
print('TRAINING COMPLETE')
print(f'Best validation accuracy: {best_acc:.1f}%')
print("="*70)

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses['train'], label='Training')
plt.plot(losses['val'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(accs['train'], label='Training')
plt.plot(accs['val'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.tight_layout()
plt.show()


#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
net.load_state_dict(torch.load('steer_net.pth', map_location = device))
net.to(device)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in valloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        
        # the class with the highest energy (logit) is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in val_ds.class_labels}
total_pred = {classname: 0 for classname in val_ds.class_labels}

# again no gradients needed
actual = []
predicted = []
with torch.no_grad():
    for data in valloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        actual += labels.detach().cpu().tolist()
        predicted += predictions.detach().cpu().tolist()

        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[val_ds.class_labels[label.item()]] += 1
            total_pred[val_ds.class_labels[label.item()]] += 1

cm = metrics.confusion_matrix(actual, predicted, normalize = 'true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=val_ds.class_labels)
disp.plot()
plt.show()

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')