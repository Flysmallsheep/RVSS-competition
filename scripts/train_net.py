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

#######################################################################################################################################
####     This tutorial is adapted from the PyTorch "Train a Classifier" tutorial                                                   ####
####     Please review here if you get stuck: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html                   ####
#######################################################################################################################################
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
BATCH_SIZE = 2048


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

from torch.utils.data import WeightedRandomSampler
import numpy as np

def make_weighted_sampler(dataset, num_classes=5):
    # Get label for every sample in the dataset
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y))
    labels = np.array(labels)

    # Count class frequency
    class_counts = np.bincount(labels, minlength=num_classes)
    print("Train class counts:", class_counts.tolist())

    # Inverse frequency weights (avoid div-by-zero)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]

    # Sample class 'straight' (class 2) 1.1x more than the others
    sample_weights[labels == 2] *= 1.1

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),   # samples per epoch
        replacement=True                  # oversampling works via replacement
    )
    return sampler


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

train_sampler = make_weighted_sampler(train_ds, num_classes=5)

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

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(1344, 256)
#         self.fc2 = nn.Linear(256, 5)

#         self.relu = nn.ReLU()


#     def forward(self, x):
#         #extract features with convolutional layers
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
        
#         #linear layer for classification
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
       
#         return x

###########################################################
##########  MobileNet V3 Small Model             ##########
###########################################################
class Net(nn.Module):
    def __init__(self, num_classes=5, pretrained=False, dropout=0.2, freeze_backbone=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v3_small(weights=weights)

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            for p in self.model.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)

net = Net().to(device)

#######################################################################################################################################
####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
#######################################################################################################################################

#for classification tasks
criterion = nn.CrossEntropyLoss()
#You could use also ADAM
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.01)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,  # every 5 epochs
    gamma=0.5     # halve the LR
)


#######################################################################################################################################
####     TRAINING LOOP                                                                                                             ####
#######################################################################################################################################
num_epochs = 15
losses = {'train': [], 'val': []} # Stores average loss per epoch for training and validation
accs = {'train': [], 'val': []}   # Stores average accuracy per epoch for training and validation
best_acc = 0 # Keeps track of the best validation accuracy seen so far (used to decide when to save the model).
for epoch in range(num_epochs):  # loop over the dataset multiple times

    epoch_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0): # yields mini-batches
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # reset the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item() # Adds the scalar loss for this batch to the epoch total.

        _, predicted = torch.max(outputs, 1) # Finds the class with the highest logit for each sample in the batch.
        total += labels.size(0) # Adds number of samples in this batch.
        correct += (predicted == labels).sum().item() # Counts number of correct predictions in this batch.

    print(f'Epoch {epoch + 1} loss: {epoch_loss / len(trainloader)}')
    losses['train'] += [epoch_loss / len(trainloader)] # mean loss over all batchesper epoch.
    accs['train'] += [100.*correct/total]  # accuracy per epoch.
 
    # Validation Setup:
    correct_pred = {classname: 0 for classname in val_ds.class_labels} # e.g. correct_pred['left'] = how many of those were predicted correctly
    total_pred = {classname: 0 for classname in val_ds.class_labels} # e.g. total_pred['left'] = how many “left” examples exist in validation

    # Validation Loop:
    # Question: how does validation data ensure random sampling of the dataset? where in the code is this ensured? Or the training data is located in a different order to the validation data?
    val_loss = 0
    with torch.no_grad(): # disable gradient computation to save memory and speed up inference.
        for data in valloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1) # Finds the class with the highest logit for each sample in the batch, and returns the index of the class.
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[val_ds.class_labels[label.item()]] += 1
                total_pred[val_ds.class_labels[label.item()]] += 1

    # print accuracy for each class
    class_accs = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname] # per-class accuracy.
        class_accs += [accuracy]

    accs['val'] += [np.mean(class_accs)] # mean accuracy over all classes.
    losses['val'] += [val_loss/len(valloader)] # mean loss over all batches.
    print(f"Val loss: {val_loss}")
    scheduler.step()
    print("LR:", optimizer.param_groups[0]["lr"])

    if np.mean(class_accs) > best_acc: # save the model if the current validation accuracy is better than the best seen so far.
        torch.save(net.state_dict(), 'steer_net.pth') # So at the end, steer_net.pth is the best-performing model during training (according to validation accuracy).
        best_acc = np.mean(class_accs)

print('Finished Training')

plt.plot(losses['train'], label = 'Training')
plt.plot(losses['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(accs['train'], label = 'Training')
plt.plot(accs['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
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