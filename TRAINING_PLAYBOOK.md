# Training & Validation Playbook

## Using train_starter and val_starter Datasets

This playbook provides a complete step-by-step guide to train and validate your CNN model using the provided starter datasets.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Dataset Setup](#2-dataset-setup)
3. [Understanding the Data](#3-understanding-the-data)
4. [Training the Model](#4-training-the-model)
5. [Monitoring Training](#5-monitoring-training)
6. [Evaluating Results](#6-evaluating-results)
7. [Saving & Loading Models](#7-saving--loading-models)
8. [Troubleshooting](#8-troubleshooting)
9. [Next Steps](#9-next-steps)

---

## 1. Prerequisites

### Check Your Environment

First, verify that your environment is set up correctly:

```bash
# Navigate to the repository root
cd /Users/kevinyang/Desktop/RVSS/RVSS_Need4Speed

# Check if dependencies are installed
pixi run test
```

Expected output:

- ✅ All Python packages imported successfully
- ✅ Dataset found and complete (or instructions to download)

### Activate the Environment

```bash
# Activate pixi environment (CPU version)
pixi shell

# OR if you have CUDA GPU support
pixi shell -e cuda
```

---

## 2. Dataset Setup

### Step 2.1: Download the Starter Dataset

The repository provides pre-collected training and validation data. Download it:

```bash
# Download and extract the dataset
pixi run get_dataset
```

**What this does:**

- Downloads `data.zip` from HuggingFace
- Extracts to `data/train_starter/` and `data/val_starter/`
- Creates 793 training images and 436 validation images

### Step 2.2: Verify Dataset Structure

Check that the data folders exist:

```bash
# Check training data
ls data/train_starter/ | head -5
# Should show files like: 0000000.00.jpg, 0000001.30.jpg, etc.

# Check validation data
ls data/val_starter/ | head -5
# Should show similar format

# Count files
echo "Training images: $(ls data/train_starter/*.jpg | wc -l)"
echo "Validation images: $(ls data/val_starter/*.jpg | wc -l)"
```

**Expected output:**

- Training images: 793
- Validation images: 436

### Step 2.3: Understand Data Format

Each image filename encodes the steering angle:

- Format: `{6-digit-number}{steering_angle}.jpg`
- Example: `0000140.50.jpg` = image #140 with steering angle 0.50 (sharp right)

**Steering angle ranges:**

- `-0.5` to `-0.01`: Sharp left / Left
- `0.00`: Straight
- `0.01` to `0.5`: Right / Sharp right

---

## 3. Understanding the Data

### Step 3.1: Inspect the Dataset

Before training, let's understand what we're working with. The training script will show you:

1. **Dataset size**: Number of images
2. **Label distribution**: How many examples per class
3. **Sample images**: Visual examples with labels

### Step 3.2: Check Label Balance

The training script automatically shows label distribution. Look for:

- **Balanced dataset**: Roughly equal examples per class
- **Imbalanced dataset**: One class dominates (e.g., 80% "straight")

**If imbalanced**, consider:

- Collecting more data for underrepresented classes
- Using class weights in loss function
- Data augmentation

---

## 4. Training the Model

### Step 4.1: Review Training Configuration

The training script (`scripts/train_net.py`) is already configured to use starter data:

```python
# Training dataset
train_ds = SteerDataSet('data/train_starter', '.jpg', transform)

# Validation dataset
val_ds = SteerDataSet('data/val_starter', '.jpg', transform)
```

**Key hyperparameters** (in `train_net.py`):

- **Epochs**: 10 (line 150)
- **Batch size**: 8 (line 52)
- **Learning rate**: 0.001 (line 141)
- **Optimizer**: SGD with momentum 0.9
- **Loss function**: CrossEntropyLoss

### Step 4.2: Start Training

Run the training script:

```bash
# From repository root
pixi run python scripts/train_net.py

# OR if you're in pixi shell
python scripts/train_net.py
```

### Step 4.3: What Happens During Training

The script will:

1. **Load datasets** (train_starter and val_starter)
2. **Show data statistics**:
   - Number of images
   - Label distribution (bar chart). The current sample training and validation dataset are not evenly-distributed
   - Sample images with labels
3. **Initialize network** (CNN with 2 conv layers + 2 FC layers)
4. **Train for 10 epochs**:
   - For each epoch:
     - Train on all training images
     - Validate on all validation images
     - Print training/validation loss and accuracy
     - Save best model if validation accuracy improves
5. **Show final results**:
   - Training/validation loss curves
   - Training/validation accuracy curves
   - Confusion matrix
   - Per-class accuracy

### Step 4.4: Expected Output

You should see output like:

```
The train dataset contains 793 images
Input to network shape: torch.Size([8, 3, 40, 60])
[Shows label distribution bar chart]
[Shows sample images]

The validation dataset contains 436 images
[Shows validation label distribution]

Epoch 1 loss: 1.2345
Epoch 2 loss: 0.9876
...
Epoch 10 loss: 0.5432

Finished Training
Accuracy of the network on the 436 test images: 75%
Accuracy for class: sharp left is 70.0%
Accuracy for class: left is 72.5%
...
```

---

## 5. Monitoring Training

### Step 5.1: Watch for These Metrics

**Training Loss:**

- Should **decrease** over epochs
- If it increases or plateaus, you may need to adjust learning rate

**Validation Loss:**

- Should **decrease** (ideally tracking training loss)
- If validation loss increases while training loss decreases → **overfitting**

**Accuracy:**

- Training accuracy: Should increase
- Validation accuracy: Should increase (but may be lower than training)
- Gap between train/val accuracy indicates overfitting

### Step 5.2: Interpret the Plots

After training, you'll see:

1. **Loss curves** (Training vs Validation):
   - Both should trend downward
   - Validation should be close to training (not much higher)

2. **Accuracy curves**:
   - Both should trend upward
   - Training accuracy may be higher (normal)

3. **Confusion matrix**:
   - Shows which classes are confused with each other
   - Diagonal = correct predictions
   - Off-diagonal = misclassifications

### Step 5.3: Healthy Training Signs

✅ **Good signs:**

- Loss decreases steadily
- Validation accuracy improves
- No huge gap between train/val metrics
- Model saves (means validation accuracy improved)

❌ **Warning signs:**

- Loss not decreasing → learning rate too high/low
- Validation loss increases → overfitting
- Accuracy stuck at random chance (~20% for 5 classes) → model not learning

---

## 6. Evaluating Results

### Step 6.1: Check Saved Model

After training, check if model was saved:

```bash
# Check if model file exists
ls -lh steer_net.pth

# Should show file size (typically a few MB)
```

**Note:** Model only saves if validation accuracy improved. If you don't see `steer_net.pth`, the model didn't improve during training.

### Step 6.2: Review Final Metrics

The script prints:

- **Overall accuracy**: Percentage of correct predictions
- **Per-class accuracy**: Accuracy for each steering direction
- **Confusion matrix**: Visual representation of predictions

### Step 6.3: Analyze Per-Class Performance

Look at per-class accuracies:

- Which classes perform well?
- Which classes are confused?

**Common issues:**

- "Straight" class dominates → model always predicts straight
- Left/right confused → need more diverse training data
- Sharp turns poorly predicted → need more extreme steering examples

---

## 7. Saving & Loading Models

### Step 7.1: Model is Auto-Saved

The training script automatically saves the best model:

- **Filename**: `steer_net.pth`
- **Location**: Repository root (same directory as training script)
- **When**: Only when validation accuracy improves

### Step 7.2: Load Model for Inference

To use the trained model later (e.g., in `deploy.py`):

```python
import torch
from train_net import Net  # Import your network class

# Initialize network
net = Net()

# Load saved weights
net.load_state_dict(torch.load('steer_net.pth'))

# Set to evaluation mode (important!)
net.eval()

# Now you can use it for predictions
with torch.no_grad():
    output = net(image_tensor)
    predicted_class = torch.max(output, 1)[1]
```

### Step 7.3: Save Additional Checkpoints (Optional)

To save models at specific epochs or with custom names:

```python
# In training loop, add:
if epoch % 5 == 0:  # Save every 5 epochs
    torch.save(net.state_dict(), f'steer_net_epoch_{epoch}.pth')
```

---

## 8. Troubleshooting

### Problem: "Dataset not found"

**Solution:**

```bash
# Download the dataset
pixi run get_dataset

# Verify it exists
ls data/train_starter/
ls data/val_starter/
```

### Problem: "CUDA out of memory"

**Solution:**

- Reduce batch size in `train_net.py` (line 52): `batch_size=4` or `batch_size=2`
- Use CPU instead: `pixi shell` (not `pixi shell -e cuda`)

### Problem: Loss not decreasing

**Possible causes:**

1. Learning rate too high → try `lr=0.0001` (line 141)
2. Learning rate too low → try `lr=0.01`
3. Model architecture too simple → add more layers
4. Data not loading correctly → check dataset paths

**Solution:**

```python
# In train_net.py, modify optimizer:
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)  # Lower LR
# OR
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Try Adam optimizer
```

### Problem: Overfitting (validation loss increases)

**Solutions:**

1. **Add dropout** to network:

   ```python
   self.dropout = nn.Dropout(0.5)  # In __init__
   x = self.dropout(x)  # Before fc2 in forward
   ```

2. **Reduce model complexity** (fewer parameters)

3. **Data augmentation** (flip images, adjust brightness, etc.)

4. **Early stopping**: Stop training when validation loss stops improving

### Problem: Model always predicts same class

**Cause:** Imbalanced dataset (e.g., 80% "straight" examples)

**Solution:**

1. Collect more diverse data
2. Use class weights in loss:
   ```python
   # Calculate class weights
   from torch.utils.data import WeightedRandomSampler
   # ... (implement weighted sampling)
   ```

### Problem: Training is very slow

**Solutions:**

- Use GPU: `pixi shell -e cuda`
- Increase batch size (if memory allows): `batch_size=16`
- Reduce image size in transforms: `transforms.Resize((30, 45))`

### Problem: "ModuleNotFoundError: No module named 'steerDS'"

**Solution:**

```bash
# Make sure you're running from scripts directory OR
# Add scripts to Python path
cd scripts
python train_net.py
```

---

## 9. Next Steps

### After Successful Training

1. **Test on Robot**:
   - Use `deploy.py` to test your model on the actual robot
   - Collect real-world data and fine-tune if needed

2. **Improve Model**:
   - Experiment with different architectures
   - Try different hyperparameters
   - Add data augmentation

3. **Optimize for Competition**:
   - Test on actual track tiles
   - Fine-tune with competition-specific data
   - Implement stop sign detection (optional)

### Experiment Ideas

1. **Modify Network Architecture**:
   - Add more convolutional layers
   - Try different activation functions
   - Add batch normalization

2. **Hyperparameter Tuning**:
   - Learning rate: 0.0001, 0.001, 0.01
   - Batch size: 4, 8, 16, 32
   - Optimizer: SGD, Adam, RMSprop
   - Epochs: 10, 20, 50

3. **Data Augmentation**:
   - Horizontal flips (reverse steering angle)
   - Brightness/contrast adjustments
   - Small rotations

4. **Advanced Techniques**:
   - Transfer learning (use pre-trained ResNet, etc.)
   - Learning rate scheduling
   - Ensemble models

---

## 📊 Quick Reference Commands

```bash
# Setup
pixi run get_dataset                    # Download dataset
pixi run test                           # Check installation

# Training
pixi run python scripts/train_net.py   # Train model

# Check results
ls -lh steer_net.pth                    # Verify model saved
python -c "import torch; print(torch.load('steer_net.pth').keys())"  # Inspect model

# Clean up (if needed)
rm steer_net.pth                        # Delete old model
```

---

## 📝 Training Checklist

Before training:

- [ ] Environment activated (`pixi shell`)
- [ ] Dataset downloaded (`pixi run get_dataset`)
- [ ] Dataset verified (793 train, 436 val images)
- [ ] Training script reviewed (`scripts/train_net.py`)

During training:

- [ ] Training loss decreases
- [ ] Validation accuracy improves
- [ ] Model file created (`steer_net.pth`)
- [ ] No errors in output

After training:

- [ ] Review loss/accuracy curves
- [ ] Check confusion matrix
- [ ] Analyze per-class performance
- [ ] Model ready for deployment

---

## 🎯 Success Criteria

Your training is successful if:

- ✅ Training loss decreases over epochs
- ✅ Validation accuracy > 60% (baseline)
- ✅ Model file `steer_net.pth` exists
- ✅ No major class imbalance in predictions
- ✅ Confusion matrix shows reasonable diagonal values

**Good baseline performance:**

- Overall accuracy: 70-80%
- Per-class accuracy: 60-90% (varies by class)

**Excellent performance:**

- Overall accuracy: >85%
- Balanced per-class accuracy: >75% for all classes

---

## 💡 Tips for Best Results

1. **Start simple**: Use the provided architecture first, then experiment
2. **Monitor closely**: Watch the first few epochs to catch issues early
3. **Save often**: Consider saving checkpoints at multiple epochs
4. **Document changes**: Note what hyperparameters you tried
5. **Visualize**: Look at the confusion matrix to understand failures
6. **Iterate**: Training is iterative - don't expect perfect results first time

---

## 📚 Additional Resources

- **PyTorch Tutorial**: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- **Network Training Guide**: `instructions/NetworkTraining.md`
- **Deployment Guide**: `instructions/NetworkDeployed.md`

---

Training1:
Accuracy of the network on the 436 test images: 64 %
Accuracy for class: sharp left is 73.1%
Accuracy for class: left is 79.2%
Accuracy for class: straight is 49.4%
Accuracy for class: right is 69.7%
Accuracy for class: sharp right is 60.0%

Analysis on the loss / accuracy curve: The model overfitted.

Change:

1. Add dropout 0.5
2. Add L2 regularization: weight_decay=1e-4

Training2:
Accuracy of the network on the 436 test images: 63 %
Accuracy for class: sharp left is 73.1%
Accuracy for class: left is 53.8%
Accuracy for class: straight is 69.5%
Accuracy for class: right is 61.8%
Accuracy for class: sharp right is 60.0%

Chance: increase the Epoch from 10 to 30.

Training3:
Accuracy of the network on the 436 test images: 61 %
Accuracy for class: sharp left is 69.2%
Accuracy for class: left is 56.9%
Accuracy for class: straight is 71.4%
Accuracy for class: right is 53.9%
Accuracy for class: sharp right is 56.0%

Analysis: The model overfitted.

Change:

1. Change dropout from 0.5 to 0.7
2. Change weight_decay from 1e-4 to 1e-3

Training4:
Accuracy of the network on the 436 test images: 55 %
Accuracy for class: sharp left is 84.6%
Accuracy for class: left is 55.4%
Accuracy for class: straight is 57.1%
Accuracy for class: right is 59.2%
Accuracy for class: sharp right is 66.0%

Analysis: Worse result, reverse the previous change

Change:

1. Revert dropout from 0.7 to 0.5
2. Revert weight_decay from 1e-3 to 1e-4

Training5:
Accuracy of the network on the 436 test images: 61 %
Accuracy for class: sharp left is 69.2%
Accuracy for class: left is 56.9%
Accuracy for class: straight is 71.4%
Accuracy for class: right is 53.9%
Accuracy for class: sharp right is 56.0%

Analysis: it's a small model with small dataset, the regularization (dropout and weight_decay) may overkill.

Change: remove all regularization (dropout and weight_decay)

---

Potential levers:

0. Collect more, diverse data (2000+) and keep the class balanced (most important)
1. Data Augmentation (most important):
   1. high-resolution image -> down-sample the image to make the network run more efficiently
   2. crop the images to bottom half that contains the track, also change the size of self.fc1
   3. flip images horizontally, as well as the steering angle, to create more data?
   4. brightness, contrast, flipping, rotation. What are more data augmentation methods?
2. modify learning rate, different optimizer (Adam)
3. Add more layers? or add more neurons? change the architecture?
   1. instead of hard argmax → class → fixed angle, use soft probabilities:
      Network outputs logits for 5 classes
      Convert to probabilities P_k
      Map each class to an angle A_k (e.g. [-0.5, -0.25, 0, 0.25, 0.5])
      Output steering probabilities - that gives continuous-ish control
   2. Increase bin to 9 or 15?
   3. Move to regression only if you need more precision/speed provided you have lots of data
4. Regularization: Different dropout and weight_decay levels, BatchNorm (Note: this CNN might be too small to regularize

---

Pre-run checklist:

1. If you change SteerDS.py or train_net.py (layers, sizes), you must mirror it in deploy.py
   1. network
   2. Data processing
