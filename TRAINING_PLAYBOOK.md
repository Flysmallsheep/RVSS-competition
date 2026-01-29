# RVSS Need4Speed — Training & Deployment Runbook

Goal: run an **end-to-end loop** that is repeatable and safe:
**collect → clean/augment → train → deploy** (optional: stop sign detection).

---

## Quickstart (environment + starter data)

```bash
pixi install
pixi run test
pixi run get_dataset
```

Starter data should appear at:

- `data/train_starter/` (793 images)
- `data/val_starter/` (436 images)

---

## 1) Data collection (robot) — step-by-step

### 1.1 Prep

- Build a track with **variety**: straights, gentle curves, tight curves.
- Plan for **balanced steering**: don’t collect only “straight”.
- Start slow; increase speed only after stability.

### 1.2 Create output folders (per session)

Use a new folder each session so cleanup/reverts are easy:

```bash
mkdir -p data/train_session1 data/val_session1
```

### 1.3 Run collection

`scripts/collect.py` teleops the robot and saves an image every timestep.

```bash
python scripts/collect.py --ip <robot_ip> --folder train_session1
```

Controls:

- Left / Right arrows: steer more left/right (step changes)
- Up/Down: reset to straight (angle=0)
- Space: stop + exit

### 1.4 Avoid overwriting runs

If you re-run into the same folder, pass `--im_num` to continue numbering:

```bash
python scripts/collect.py --ip <robot_ip> --folder train_session1 --im_num 500
```

### 1.5 Collect validation data (must be “unseen”)

Validation should be a **different drive** (avoid leakage):

- different lap / different ordering / different lighting if possible

```bash
python scripts/collect.py --ip <robot_ip> --folder val_session1
```

### 1.6 Data format (what the code expects)

Filenames encode steering angle:

- `0000140.50.jpg` → index 140, steering angle +0.50

The dataset loader (`scripts/steerDS.py`) parses steering from filename and converts it into 5 classes.

---

## 2) Before training — checklist (do every run)

### Data checks

- [ ] Train/val folders exist and are not empty
- [ ] Train and val are **different drives** (no leakage)
- [ ] No obvious junk frames (blurred camera, off-track, hand covering camera)
- [ ] Class balance is reasonable (avoid one dominant class)

Quick class-balance check (rough): run `python scripts/train_net.py` and look at the label histogram it plots.

### Pipeline consistency checks

- [ ] `steerDS.py` crop matches your intent (track region)
- [ ] Train transforms include augmentation; val transforms do **not**
- [ ] If you changed resize/crop/architecture: `fc1` input size still matches
- [ ] Deployment will use **the same** crop/resize/normalize and color order (BGR vs RGB)

### Stop sign detection (optional) checks to remember (from `instructions/StopSignDetection.md`)

- [ ] Stop sign can be on urban or rural tiles → thresholds must be robust
- [ ] `cv2` uses BGR; machinevision-toolbox expects RGB by default
- [ ] `blobs()` may throw when no blobs → wrap with try/except

---

## 3) Transforms & augmentation (what, why, where)

Rule: **augment training only**; validation must be deterministic.

### 3.1 Resize (down-sample)

Purpose: faster training/inference; often enough detail for steering.

Where: `scripts/train_net.py` transform.

```python
transforms.Resize((40, 60))  # (H, W)
```

If you change resize, you may need to update `fc1` (see 3.4).

### 3.2 Crop to track region

Purpose: remove sky/walls; focus on road geometry.

Where: `scripts/steerDS.py` currently:

```python
img = cv2.imread(f)[120:, :, :]
```

Tip: keep the _final network input size_ fixed (via `Resize`) even if you change crop.

### 3.3 Augmentation: brightness/contrast (safe default)

Purpose: robustness to lighting changes.

Where: `scripts/train_net.py` (train transform only).

```python
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((40, 60)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((40, 60)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

Then pass into datasets:

```python
train_ds = SteerDataSet(..., transform=train_transform, augment_flip=False)
val_ds   = SteerDataSet(..., transform=val_transform,   augment_flip=False)
```

### 3.4 Augmentation: horizontal flip + steering flip (high impact)

Purpose: doubles effective data; left/right symmetry; helps balancing turns.

Where: implement in `scripts/steerDS.py` so image and label are flipped together.

```python
# scripts/steerDS.py
import random

class SteerDataSet(Dataset):
    def __init__(self, root_folder, img_ext=\".jpg\", transform=None, augment_flip=False):
        ...
        self.augment_flip = augment_flip

    def __getitem__(self, idx):
        f = self.filenames[idx]
        img = cv2.imread(f)[120:, :, :]

        steering = path.split(f)[-1].split(self.img_ext)[0][6:]
        steering = float(steering)

        if self.augment_flip and random.random() > 0.5:
            img = cv2.flip(img, 1)   # horizontal flip
            steering = -steering     # MUST flip steering too

        # then apply transform, then map steering -> class as usual
        ...
```

In `scripts/train_net.py`:

```python
train_ds = SteerDataSet(..., transform=train_transform, augment_flip=True)
val_ds   = SteerDataSet(..., transform=val_transform,   augment_flip=False)
```

### 3.5 `fc1` size warning (when you change input/architecture)

From `instructions/NetworkTraining.md`: `fc1` depends on input size and conv/pool stack.

Practical method:

- print the tensor shape right before `fc1` for a single batch and update `fc1` accordingly.

---

## 4) Train (baseline)

### 4.1 Train on starter data

```bash
pixi run python scripts/train_net.py
```

### 4.2 Train on your own data

Point `train_net.py` to your folders (edit the dataset paths):

- `data/train_session1`
- `data/val_session1`

---

## 5) Potential levers (high → low impact)

### A) Data (highest impact)

- Collect more diverse data (target 2k+ images): tight turns + transitions.
- Keep classes balanced (don’t let “straight” dominate).
- Val set must be unseen (different drive).

### B) Preprocessing / augmentation (high impact)

- Train-only `ColorJitter` (lighting robustness).
- Horizontal flip + steering negation (best “free” improvement).
- Crop to track region.
- Resize tradeoff: smaller = faster; larger = more detail (keep deploy consistent).

### C) Model (medium impact)

- Underfitting → increase capacity (more conv channels / add a conv layer).
- Overfitting → light regularization (start small; don’t stack heavy dropout+L2 on small data).

### D) Optimization (medium impact)

- Try Adam vs SGD.
- Tune learning rate (most sensitive hyperparameter).

### E) Deployment behavior (often overlooked)

- Prefer soft probabilities → expected steering angle (smoother than argmax).
- Add temporal smoothing (moving average / majority vote).

---

## 6) Deployment pre-run checklist

### Safety + setup

- [ ] First test on blocks / wheels off ground
- [ ] Start with conservative speeds (`Kd`, `Ka`)
- [ ] Kill plan: Ctrl+C / spacebar / power
- [ ] Battery OK; Wi‑Fi stable

### Code consistency (most common failure mode)

- [ ] `deploy.py` uses **the same** network architecture as training
- [ ] `deploy.py` uses **the same** preprocessing as training:
  - crop rows
  - resize H×W
  - normalize mean/std
  - color order (BGR vs RGB)
- [ ] `deploy.py` calls `net.eval()` and uses `torch.no_grad()`
- [ ] `steer_net.pth` path is correct and exists

### Behavior checks

- [ ] Angle output is clamped to expected range (e.g. `[-0.5, 0.5]`)
- [ ] Turn direction sanity: sign(angle) produces the correct turn on your robot
- [ ] Optional smoothing enabled if steering jitters

### Stop sign detection (optional)

- [ ] Works on both urban and rural tiles
- [ ] No-blob case handled (no crash)
- [ ] BGR↔RGB handled correctly

---

## 7) Clean experiment logbook (fill this in each run)

```text
Run ID:
Date/time:
Branch/commit:

Dataset:
  - train folder(s):
  - val folder(s):
  - counts (train/val):
  - class balance notes:

Preprocessing:
  - crop:
  - resize:
  - normalize:
  - augmentations (train only):

Model:
  - architecture summary:
  - output type: classification (K=?) / regression

Training:
  - optimizer + lr:
  - batch size:
  - epochs:
  - regularization (dropout/weight_decay/etc):

Results:
  - val overall acc (%):
  - per-class acc:
  - notes (what fails? straights? tight turns?):

Decision / next action:
  - keep / revert / change:
  - next run hypothesis:
```
