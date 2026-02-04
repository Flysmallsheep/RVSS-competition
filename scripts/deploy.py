#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

# Import stop sign detector
from stop_sign_detector import StopSignDetector


parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument(
    '--debug_stop',
    action='store_true',
    help='Save frames + red mask + stats when stop sign triggers; log area during inference for tuning min_area',
)
parser.add_argument(
    '--debug_stop_dir',
    type=str,
    default=None,
    help='Directory for debug images (default: deploy_debug/ in project root)',
)
args = parser.parse_args()

# Class labels (must match steerDS.py)
CLASS_LABELS = ["sharp_left", "left", "straight", "right", "sharp_right"]

bot = PiBot(ip=args.ip)

# stop the robot 
bot.setVelocity(0, 0)

#INITIALISE NETWORK HERE
###########################################################
##########  Original CNN Model                   ##########
###########################################################
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Mirror the training architecture exactly so weights load correctly.
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(1344, 256)
#         self.fc2 = nn.Linear(256, 5)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)
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


# Select CPU/GPU for inference. GPU is optional but faster if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

#LOAD NETWORK WEIGHTS HERE
# Use the same weights produced by scripts/train_net.py (default: steer_net.pth).
model_path = os.path.abspath(os.path.join(script_path, "..", "steer_net_mobile_net.pth"))
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found at {model_path}. "
        "Train first (python scripts/train_net.py) or update model_path."
    )

# Load weights onto the selected device and set eval mode to disable dropout.
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()

# Image preprocessing must match training (resize + normalize). We keep BGR order
# because training data came from cv2.imread (also BGR), so this stays consistent.
transform = transforms.Compose([
    transforms.ToTensor(),               
    transforms.Resize((40, 60)),        
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ============================================================================
# STOP SIGN DETECTOR SETUP
# ============================================================================
stop_detector = StopSignDetector(
    min_area=200,  # Tune this based on testing
)

# Stop sign handling state
stop_sign_handled = False  # True if we've already stopped for the current sign
STOP_DURATION = 2.0        # How long to stop (seconds)
RESUME_COOLDOWN = 3.0      # Cooldown after stopping before detecting again
last_stop_time = 0         # Timestamp of last stop (for cooldown)

########################################
# Debug output for stop sign tuning
########################################
debug_stop_enabled = args.debug_stop
debug_stop_dir = args.debug_stop_dir
if debug_stop_enabled:
    if debug_stop_dir is None:
        debug_stop_dir = os.path.abspath(os.path.join(script_path, "..", "deploy_debug"))
    os.makedirs(debug_stop_dir, exist_ok=True)
    debug_stop_frame_count = 0  # Number of "stop triggered" frames saved
    last_debug_log_time = 0     # Throttle inference-time area logging
    DEBUG_LOG_INTERVAL = 0.5     # Log area at most every 0.5 s when area > 0
    if debug_stop_dir:
        print(f"[debug_stop] Saving stop-sign frames and stats to: {debug_stop_dir}")

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    frame_count = 0
    debug_frames_saved = 0
    while True:
        # get an image from the the robot
        im = bot.getImage()

        #TO DO: apply any necessary image transforms
        # Crop top of the image (sky/ceiling) to focus on track like in steerDS.py.
        # Make sure the image is large enough before cropping.
        if im is None or im.shape[0] <= 120:
            continue  # skip this frame if image is invalid
        im_cropped = im[120:, :, :]

        # Apply the exact same transform pipeline as during training.
        # Add a batch dimension (1, C, H, W) because the network expects batches.
        input_tensor = transform(im_cropped).unsqueeze(0).to(device)

        #TO DO: pass image through network get a prediction
        # Run inference with gradients off (faster and lower memory).
        with torch.no_grad():
            outputs = net(input_tensor)  # logits shape: (1, 5)

        #TO DO: convert prediction into a meaningful steering angle
        # Convert class logits to probabilities for smoother steering.
        probs = torch.softmax(outputs, dim=1).squeeze(0)  # shape: (5,)

        # Map class probabilities to steering angles (bin centers).
        # These should be consistent with your labeling scheme in steerDS.py.
        class_angles = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5], device=device)
        angle = float(torch.sum(probs * class_angles).item())

        # Safety clamp to the expected range.
        angle = float(np.clip(angle, -0.5, 0.5))

        # ====================================================================
        # STOP SIGN DETECTION
        # ====================================================================
        # Check for stop signs using the full image (not cropped).
        # The stop sign could be anywhere in the frame, not just the bottom.
        
        current_time = time.time()
        
        ##################################################
        # Use the full image for stop sign detection
        ##################################################
        if debug_stop_enabled:
            stop_details = stop_detector.detect_with_details(im)
            stop_detected = stop_details['detected']
            stop_area = stop_details['largest_area']
        else:
            stop_detected, stop_area = stop_detector.detect(im)
            stop_details = None
        
        # During inference: log red blob area periodically for tuning min_area
        if debug_stop_enabled and stop_area > 0:
            if current_time - last_debug_log_time >= DEBUG_LOG_INTERVAL:
                last_debug_log_time = current_time
                triggered = "TRIGGER" if stop_detected else "below threshold"
                print(f"[stop_sign] area={stop_area} (min_area={stop_detector.min_area}) -> {triggered}")
        
        # Only handle stop sign if:
        # 1. A stop sign is detected AND close enough (area >= min_area)
        # 2. We haven't just handled a stop sign (cooldown period)
        # 3. We're not currently in the middle of handling one
        if stop_detected and not stop_sign_handled:
            # Check cooldown (don't stop again immediately after resuming)
            if current_time - last_stop_time > RESUME_COOLDOWN:
                print(f"STOP SIGN DETECTED! Area: {stop_area}. Stopping...")
                
                ################################################################################
                # Save frame, mask, and overlay when --debug_stop is on (for tuning min_area)
                ##########################################################################################
                if debug_stop_enabled and stop_details is not None:
                    debug_stop_frame_count += 1
                    prefix = os.path.join(debug_stop_dir, f"stop_{debug_stop_frame_count:04d}")
                    cv2.imwrite(f"{prefix}_frame.jpg", im)
                    if stop_details.get('mask') is not None:
                        cv2.imwrite(f"{prefix}_mask.jpg", stop_details['mask'])
                    vis = im.copy()
                    if stop_details.get('bounding_box'):
                        x, y, w, h = stop_details['bounding_box']
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if stop_details.get('centroid'):
                        cv2.circle(vis, stop_details['centroid'], 5, (0, 0, 255), -1)
                    cv2.putText(
                        vis, f"area={stop_area} min_area={stop_detector.min_area}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    cv2.imwrite(f"{prefix}_overlay.jpg", vis)
                    print(f"[debug_stop] Saved {prefix}_frame.jpg, _mask.jpg, _overlay.jpg")
                    print(f"  largest_area={stop_area}, min_area={stop_detector.min_area}, centroid={stop_details.get('centroid')}, bbox={stop_details.get('bounding_box')}, all_areas={stop_details.get('all_areas', [])}")
                
                # STOP THE ROBOT COMPLETELY
                bot.setVelocity(0, 0)
                
                # Wait for the required duration (rules say must come to complete stop)
                time.sleep(STOP_DURATION)
                
                # Mark that we've handled this stop sign
                stop_sign_handled = True
                last_stop_time = time.time()
                
                print("Resuming...")
                continue  # Skip this iteration, resume driving on next loop
        
        # Reset the handled flag once we no longer see the stop sign
        # This allows us to detect and stop at the NEXT stop sign
        if not stop_detected and stop_sign_handled:
            # Add a small delay before resetting to avoid flickering
            if current_time - last_stop_time > RESUME_COOLDOWN:
                stop_sign_handled = False
        
        # ====================================================================
        # MOTOR CONTROL
        # ====================================================================
        Kd = 25  # Base wheel speeds, increase to go faster, decrease to go slower
        Ka = 25  # How fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
