# Data Augmentation: Explanation & Code

This document explains the three data-augmentation ideas from the playbook (lines 648–650) and gives the purpose and code for each.

**Summary**

| # | Idea | Purpose | Where |
|---|------|--------|-------|
| 1 | Down-sample high-res image | Faster training/inference, less memory; resolution still enough for steering | `train_net.py` transform: `Resize((40, 60))` |
| 2 | Crop to bottom half | Keep only track region; less sky/noise. Changing crop/resize changes flattened size → must set `fc1` to match | `steerDS.py`: `img[120:, :, :]`; `train_net.py`: `fc1` input size |
| 3 | Horizontal flip + flip steering | Double effective data; left/right symmetry; balance left/right turns | `steerDS.py`: random flip + negate angle; `train_net.py`: `augment_flip=True` for train only |

---

## 1. Down-sample high-resolution images

### Purpose

- **Speed**: Smaller images → fewer pixels → faster forward/backward passes and less memory.
- **Efficiency**: The network doesn’t need full resolution to learn “road vs turn”; medium resolution is enough.
- **Regularization**: Slight loss of detail can reduce overfitting to irrelevant high‑frequency noise.

You’re not throwing away the “task”; you’re keeping a resolution that’s sufficient for steering.

### Current code (already in `train_net.py`)

Down-sampling is already done via `Resize` in the transform:

```python
# In scripts/train_net.py, ~line 37
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((40, 60)),   # <- Down-sample: (H, W) for the network
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

- Input from the dataset is the cropped image (e.g. after `[120:, :, :]` in `steerDS.py`).
- After this pipeline, every image is **40×60** before the CNN.
- You can make training/inference faster by using a smaller size (e.g. `(32, 48)`), or slightly better quality with a larger one (e.g. `(48, 72)`), at the cost of speed and `fc1` size (see point 2).

### Optional: different sizes

```python
# Smaller = faster, less capacity
transforms.Resize((32, 48))

# Larger = more detail, slower (and requires recomputing fc1 input size)
transforms.Resize((48, 72))
```

If you change the spatial size, you must recompute the flattened size for `fc1` (see next section).

---

## 2. Crop to the bottom half (track only) and `fc1` size

### Purpose

- **Focus on the road**: The top part of the image is often sky/walls; the bottom half contains the track and lane geometry.
- **Less distraction**: Fewer irrelevant pixels → model concentrates on what matters for steering.
- **Smaller input**: Cropping reduces height → after convs you have fewer activations → **the flattened size changes**, so **`fc1` must match**.

So: crop = “use only the part that matters” + “smaller feature map” → you must set `self.fc1` to that new size.

### Current code (already in `steerDS.py`)

Cropping is already applied when loading the image:

```python
# In scripts/steerDS.py, ~line 38
img = cv2.imread(f)[120:, :, :]   # Crop: skip top 120 rows, keep rest (bottom part)
```

- `120:` = drop top 120 pixels, keep from row 120 to the end.
- If the original image height is 240, you keep 120 rows (bottom half). If height differs, the “half” is only approximate; the important thing is “bottom part with track”.

### If you change crop or resize: recompute `fc1` size

The first linear layer must receive the **flattened conv output**. That size depends on:

- Input size **after** crop and resize (e.g. 40×60),
- Conv/pool layers (kernel sizes, strides, padding).

Example for the **current** `Net` (2 convs, 2 pools, input 40×60):

- After conv1 + pool: `(6, 18, 28)`
- After conv2 + pool: `(16, 7, 12)`
- Flattened: `16 * 7 * 12 = 1344` → so `fc1` should be `nn.Linear(1344, 256)`.

If you **change crop** (e.g. `[80:, :, :]` → more rows) you must **resize to the same (H, W)** as now, **or** recompute dimensions. Easiest is to keep resize fixed (e.g. 40×60) so that crop only affects what content the model sees; then `fc1` stays 1344.

If you **change resize** to e.g. `(48, 72)`:

- Recompute step-by-step:
  - Conv1 + pool: H = (48 - 5 + 1) // 2 = 22, W = (72 - 5 + 1) // 2 = 34  → (6, 22, 34)
  - Conv2 + pool: H = (22 - 5 + 1) // 2 = 9, W = (34 - 5 + 1) // 2 = 15   → (16, 9, 15)
  - Flattened: `16 * 9 * 15 = 2160` → use `nn.Linear(2160, 256)`.

**Code pattern:** compute once and use in `Net`:

```python
# Example: input (40, 60), 2 conv layers as in current Net
# After two (conv 5x5 + pool 2x2): channels 16, H=7, W=12
fc1_input_size = 16 * 7 * 12   # 1344
self.fc1 = nn.Linear(fc1_input_size, 256)
```

So: **crop** = “use bottom half in steerDS”; **resize** = “fixed input size in transform”; **fc1** = “match the flattened size for that input size and architecture”.

---

## 3. Horizontal flip + flip steering angle

### Purpose

- **More data for free**: Every image has a mirror version; flipping doubles the number of training variants without collecting new drives.
- **Left/right symmetry**: A “turn left” scene looks like “turn right” when flipped; the label must be flipped too so the model learns “left vs right” correctly.
- **Balanced turns**: If your dataset has more left turns than right (or vice versa), flipping balances the distribution.

So: flip image **and** negate steering angle (and then map to classes again if you use classification).

### Code

You have two places to implement this:

- **Option A – in the Dataset (`steerDS.py`)**: when loading, randomly flip and negate angle.
- **Option B – in the transform**: use `RandomHorizontalFlip` and a custom transform that also flips the label.

Below is **Option A** (all logic in one place in the dataset).

**1) In `scripts/steerDS.py`** – add random flip and angle flip:

```python
import random

class SteerDataSet(Dataset):
    def __init__(self, root_folder, img_ext=".jpg", transform=None, augment_flip=False):
        self.root_folder = root_folder
        self.transform = transform
        self.img_ext = img_ext
        self.filenames = glob(path.join(self.root_folder, "*" + self.img_ext))
        self.totensor = transforms.ToTensor()
        self.augment_flip = augment_flip  # Enable only for training
        self.class_labels = [
            'sharp left', 'left', 'straight', 'right', 'sharp right'
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = self.filenames[idx]
        img = cv2.imread(f)[120:, :, :]   # crop bottom half (unchanged)

        # Optional: horizontal flip (only for training)
        if self.augment_flip and random.random() > 0.5:
            img = cv2.flip(img, 1)   # 1 = horizontal

        if self.transform is None:
            img = self.totensor(img)
        else:
            img = self.transform(img)

        steering = path.split(f)[-1].split(self.img_ext)[0][6:]
        steering = float(steering)

        # If we flipped the image, negate the steering angle
        if self.augment_flip and getattr(self, '_last_flipped', False):
            steering = -steering
        # Track whether we flipped for this sample (see below)
        # Actually we need to flip the angle *when* we flipped the image.
        # So we need to decide flip first, then apply to both image and angle.
```

Correct pattern: **decide flip once per sample, then apply to both image and angle.** Here’s a cleaner version:

```python
def __getitem__(self, idx):
    f = self.filenames[idx]
    img = cv2.imread(f)[120:, :, :]

    steering = path.split(f)[-1].split(self.img_ext)[0][6:]
    steering = float(steering)

    # Random horizontal flip (training only)
    if self.augment_flip and random.random() > 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    if self.transform is None:
        img = self.totensor(img)
    else:
        img = self.transform(img)

    # Convert steering to class (same as before)
    if steering <= -0.5:
        steering_cls = 0
    elif steering < 0:
        steering_cls = 1
    elif steering == 0:
        steering_cls = 2
    elif steering < 0.5:
        steering_cls = 3
    else:
        steering_cls = 4

    return img, steering_cls
```

**2) In `scripts/train_net.py`** – use `augment_flip=True` for training, `False` for validation:

```python
# Training dataset: enable horizontal flip augmentation
train_ds = SteerDataSet(
    os.path.join(script_path, '..', 'data', 'train_starter'),
    '.jpg',
    transform,
    augment_flip=True
)

# Validation dataset: no augmentation (deterministic evaluation)
val_ds = SteerDataSet(
    os.path.join(script_path, '..', 'data', 'val_starter'),
    '.jpg',
    transform,
    augment_flip=False
)
```

So:
- **1) Down-sample**: already done with `Resize` in the transform; adjust size if you want faster/slower or different `fc1`.
- **2) Crop**: already done in `steerDS` with `[120:, :, :]`; if you change crop/resize or conv design, recompute and set `self.fc1` to the correct flattened size.
- **3) Flip**: add `augment_flip` in `SteerDataSet`, flip image and negate `steering` when `random.random() > 0.5`, then pass `augment_flip=True` only for the training dataset.

This gives you the purpose and the code for all three playbook items.
