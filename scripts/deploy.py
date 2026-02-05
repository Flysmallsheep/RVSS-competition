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
parser.add_argument('--show', action='store_true', help='Show live camera feed during inference (for debugging)')
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
# ROI (Region of Interest) for stop sign detection - crop to ignore periphery
parser.add_argument(
    '--roi_top',
    type=int,
    default=120,
    help='Pixels to crop from top for stop sign ROI (default: 100, removes ceiling)',
)
parser.add_argument(
    '--roi_bottom',
    type=int,
    default=None,
    help='Bottom row for stop sign ROI (default: None = full height)',
)
parser.add_argument(
    '--roi_left',
    type=int,
    default=0,
    help='Pixels to crop from left for stop sign ROI (default: 0 = no left crop)',
)
parser.add_argument(
    '--roi_right',
    type=int,
    default=None,
    help='Right column for stop sign ROI (default: None = full width)',
)
parser.add_argument(
    '--no_roi',
    action='store_true',
    help='Disable ROI cropping for stop sign detection (use full image)',
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
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Mirror the training architecture exactly so weights load correctly.
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

###########################################################
##########  MobileNet V3 Small Model             ##########
# ###########################################################
# class Net(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
#         # Using pretrained weights helps the model recognize shapes (lines) faster
#         self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
#         # Re-map the classifier head to 5 classes
#         in_features = self.model.classifier[-1].in_features
#         self.model.classifier[-1] = nn.Sequential(
#             nn.Linear(in_features, num_classes)
#         )

#     def forward(self, x):
#         return self.model(x)


# Select CPU/GPU for inference. GPU is optional but faster if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

#LOAD NETWORK WEIGHTS HERE
# Use the same weights produced by scripts/train_net.py (default: steer_net.pth).
model_path = os.path.abspath(os.path.join(script_path, "..", "final_model.pth"))
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
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((40, 60)),
                                # transforms.ColorJitter(
                                #     saturation=0.3,  # +/- 30% saturation
                                #     hue=0.05         # +/- 0.05 hue (range is [-0.5, 0.5])
                                # ),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

# ============================================================================
# STOP SIGN DETECTOR SETUP
# ============================================================================
stop_detector = StopSignDetector(
    min_area=100,          # Tight: must be close enough (stops within ~10cm)
    max_area=2000,         # Reject huge blobs (orange tiles)
    min_circularity=0.35,  # Reject elongated streaks (orange tile edges)
    min_area_relaxed=80,   # Relaxed: still requires decent-sized blob
    lower_red1_relaxed=(0, 85, 50),     # Raised S from 70 to 85 (less orange leakage)
    upper_red1_relaxed=(4, 255, 255),
    lower_red2_relaxed=(174, 85, 50),   # Raised S from 70 to 85
    upper_red2_relaxed=(180, 255, 255),
)

# Region of Interest (ROI) for stop sign detection.
# Crops the image to focus only on the track area, ignoring periphery
# (shoes, walls, ceiling, etc.). Format: (top, bottom, left, right) in pixels.
# Set to None to use full image. Adjust based on your camera placement.
# Typical robot camera is 240 (H) x 320 (W).
# Use --roi_top, --roi_bottom, --roi_left, --roi_right to tune, or --no_roi to disable.
if args.no_roi:
    STOP_SIGN_ROI = None
    print("[stop_sign] ROI disabled (--no_roi), using full image")
else:
    STOP_SIGN_ROI = {
        'top': args.roi_top,       # Crop from top (removes ceiling)
        'bottom': args.roi_bottom, # Crop from bottom (None = keep to end)
        'left': args.roi_left,     # Crop from left (removes side objects)
        'right': args.roi_right,   # Crop to right edge
    }
    print(f"[stop_sign] ROI: top={args.roi_top}, bottom={args.roi_bottom}, left={args.roi_left}, right={args.roi_right}")

# Stop sign handling state
stop_sign_handled = False  # True if we've already stopped for the current sign
STOP_DURATION = 1.0        # How long to stop (seconds)
RESUME_COOLDOWN = 2.0      # Cooldown after stopping before detecting again
last_stop_time = 0         # Timestamp of last stop (for cooldown)

# Multi-frame accumulation: require N detections in last M frames to trigger
STOP_WINDOW_SIZE = 7       # Rolling window of recent frames
STOP_MIN_HITS = 3          # Need 3+ hits in 7 frames (filters single-frame orange glitches)
stop_history = []           # Rolling list of booleans (True = red detected this frame)

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

# ============================================================================
# STARTUP FRAME CAPTURE - save first N seconds for inspection
# ============================================================================
STARTUP_CAPTURE_SECS = 5  # How many seconds to capture at launch
startup_capture_dir = os.path.abspath(os.path.join(script_path, "..", "startup_frames"))
os.makedirs(startup_capture_dir, exist_ok=True)
# Clear old frames from previous run
for old_file in os.listdir(startup_capture_dir):
    if old_file.endswith(('.jpg', '.png')):
        os.remove(os.path.join(startup_capture_dir, old_file))
print(f"[startup_capture] Saving frames for first {STARTUP_CAPTURE_SECS}s to: {startup_capture_dir}")

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
    startup_time = time.time()
    startup_capture_done = False
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
        # Temperature scaling: T > 1.0 softens overconfident predictions
        # This helps when model is confident but wrong (calibration issue)
        TEMPERATURE = 6.2  # Increase to 2.0 if still too confident
        probs = torch.softmax(outputs / TEMPERATURE, dim=1).squeeze(0)  # shape: (5,)

        # Map class probabilities to steering angles (bin centers).
        # These should be consistent with your labeling scheme in steerDS.py.
        class_angles = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5], device=device)
        angle = float(torch.sum(probs * class_angles).item())

        # Safety clamp to the expected range.
        angle = float(np.clip(angle, -0.5, 0.5))

        # ====================================================================
        # CONFIDENCE-BASED SPEED ADJUSTMENT
        # ====================================================================
        # Get the model's confidence (max probability).
        # High confidence (close to 1.0) = model is sure -> drive faster
        # Low confidence (close to 0.2 for 5 classes) = uncertain -> slow down
        confidence = float(probs.max().item())
        
        # Define confidence thresholds and speed scaling
        CONFIDENCE_HIGH = 0.7    # Above this: full speed
        CONFIDENCE_LOW = 0.35    # Below this: minimum speed
        SPEED_MIN_FACTOR = 0.4   # Minimum speed as fraction of base (40%)
        
        # Linear interpolation between min and full speed based on confidence
        if confidence >= CONFIDENCE_HIGH:
            speed_factor = 1.0
        elif confidence <= CONFIDENCE_LOW:
            speed_factor = SPEED_MIN_FACTOR
        else:
            # Linear scale between CONFIDENCE_LOW and CONFIDENCE_HIGH
            speed_factor = SPEED_MIN_FACTOR + (1.0 - SPEED_MIN_FACTOR) * \
                           (confidence - CONFIDENCE_LOW) / (CONFIDENCE_HIGH - CONFIDENCE_LOW)
        
        # Optional: print confidence for debugging (comment out in competition)
        print(f"Conf: {confidence:.2f} -> Speed: {speed_factor:.2f}")

        # ====================================================================
        # STARTUP FRAME CAPTURE (first N seconds)
        # ====================================================================
        if not startup_capture_done:
            elapsed = time.time() - startup_time
            if elapsed <= STARTUP_CAPTURE_SECS:
                frame_count += 1
                annotated = im.copy()
                # Draw steering + confidence info
                pred_class = CLASS_LABELS[int(probs.argmax().item())]
                cv2.putText(annotated, f"angle:{angle:.2f} conf:{confidence:.2f} cls:{pred_class}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f"speed_factor:{speed_factor:.2f} t={elapsed:.1f}s",
                            (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # Draw the steering crop line (same 120px crop used for inference)
                cv2.line(annotated, (0, 120), (annotated.shape[1], 120), (255, 0, 0), 1)
                # Draw ROI box if set (for stop sign)
                if STOP_SIGN_ROI is not None:
                    h_img, w_img = im.shape[:2]
                    roi_t = STOP_SIGN_ROI.get('top', 0) or 0
                    roi_b = STOP_SIGN_ROI.get('bottom') or h_img
                    roi_l = STOP_SIGN_ROI.get('left', 0) or 0
                    roi_r = STOP_SIGN_ROI.get('right') or w_img
                    cv2.rectangle(annotated, (roi_l, roi_t), (roi_r, roi_b), (0, 0, 255), 1)
                    cv2.putText(annotated, "stop ROI", (roi_l + 2, roi_t + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                # Save raw + annotated
                fname = f"{frame_count:04d}_t{elapsed:.2f}s"
                cv2.imwrite(os.path.join(startup_capture_dir, f"{fname}_raw.jpg"), im)
                cv2.imwrite(os.path.join(startup_capture_dir, f"{fname}_annotated.jpg"), annotated)
            else:
                startup_capture_done = True
                print(f"[startup_capture] Done! Saved {frame_count} frame pairs to {startup_capture_dir}")

        # Live camera feed for debugging (press 'q' to quit)
        if args.show and im is not None:
            vis = im.copy()
            cv2.putText(
                vis,
                f"angle: {angle:.2f} conf: {confidence:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("deploy live", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ====================================================================
        # STOP SIGN DETECTION
        # ====================================================================
        # Crop image to ROI to focus on track and ignore periphery (shoes, walls, etc.)
        
        current_time = time.time()
        
        ##################################################
        # Apply ROI crop for stop sign detection
        ##################################################
        if STOP_SIGN_ROI is not None:
            h, w = im.shape[:2]
            top = STOP_SIGN_ROI.get('top', 0) or 0
            bottom = STOP_SIGN_ROI.get('bottom') or h
            left = STOP_SIGN_ROI.get('left', 0) or 0
            right = STOP_SIGN_ROI.get('right') or w
            # Clamp to valid range
            top = max(0, min(top, h))
            bottom = max(top, min(bottom, h))
            left = max(0, min(left, w))
            right = max(left, min(right, w))
            im_for_stop = im[top:bottom, left:right, :]
        else:
            im_for_stop = im  # Use full image
        
        # ====================================================================
        # TWO-TIER DETECTION: tight (sharp) + relaxed (blurred)
        # ====================================================================
        # Tight: high saturation, narrow hue — reliable on sharp frames
        # Relaxed: low saturation, wider hue — catches motion-blurred red
        both = stop_detector.detect_both(im_for_stop)
        stop_detected_tight = both['tight_detected']
        stop_detected_relaxed = both['relaxed_detected']
        stop_area = both['tight_area']
        relaxed_area = both['relaxed_area']
        
        # For debug logging, also get detailed info if needed
        stop_details = None
        if debug_stop_enabled:
            stop_details = stop_detector.detect_with_details(im_for_stop)
        
        # During inference: log both tight and relaxed areas
        if debug_stop_enabled and (stop_area > 0 or relaxed_area > 0):
            if current_time - last_debug_log_time >= DEBUG_LOG_INTERVAL:
                last_debug_log_time = current_time
                print(f"[stop_sign] tight: area={stop_area} ({'HIT' if stop_detected_tight else 'miss'}) | "
                      f"relaxed: area={relaxed_area} ({'HIT' if stop_detected_relaxed else 'miss'})")
        
        # ====================================================================
        # MULTI-FRAME ACCUMULATION (fed by relaxed detection)
        # ====================================================================
        # Relaxed detections feed the rolling window.
        # This catches blurred frames that tight thresholds would miss.
        stop_history.append(stop_detected_relaxed)
        if len(stop_history) > STOP_WINDOW_SIZE:
            stop_history.pop(0)
        
        hits = sum(stop_history)
        multi_frame_triggered = hits >= STOP_MIN_HITS
        
        # Also trigger immediately on a single tight detection (sharp frame, high confidence)
        immediate_trigger = stop_detected_tight
        
        if debug_stop_enabled and hits > 0:
            print(f"[stop_sign] rolling window: {hits}/{len(stop_history)} hits (need {STOP_MIN_HITS}) | tight={stop_detected_tight}")
        
        # Trigger stop if: tight single-frame OR enough relaxed hits in window
        if (immediate_trigger or multi_frame_triggered) and not stop_sign_handled:
            # Check cooldown (don't stop again immediately after resuming)
            if current_time - last_stop_time > RESUME_COOLDOWN:
                print(f"STOP SIGN DETECTED! Area: {stop_area}. Stopping...")
                
                ################################################################################
                # Save frame, mask, and overlay when --debug_stop is on (for tuning min_area)
                ##########################################################################################
                if debug_stop_enabled and stop_details is not None:
                    debug_stop_frame_count += 1
                    prefix = os.path.join(debug_stop_dir, f"stop_{debug_stop_frame_count:04d}")
                    # Save full frame for reference
                    cv2.imwrite(f"{prefix}_frame_full.jpg", im)
                    # Save the ROI-cropped frame (what the detector actually sees)
                    cv2.imwrite(f"{prefix}_frame_roi.jpg", im_for_stop)
                    if stop_details.get('mask') is not None:
                        cv2.imwrite(f"{prefix}_mask.jpg", stop_details['mask'])
                    # Draw overlay on the ROI image
                    vis = im_for_stop.copy()
                    if stop_details.get('bounding_box'):
                        x, y, w, h = stop_details['bounding_box']
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if stop_details.get('centroid'):
                        cv2.circle(vis, stop_details['centroid'], 5, (0, 0, 255), -1)
                    roi_str = f"ROI: top={STOP_SIGN_ROI['top']}, left={STOP_SIGN_ROI['left']}" if STOP_SIGN_ROI else "ROI: None"
                    circ_val = stop_details.get('circularity', 0)
                    cv2.putText(
                        vis, f"area={stop_area} circ={circ_val:.2f} min={stop_detector.min_area}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                    cv2.putText(
                        vis, roi_str,
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                    )
                    cv2.imwrite(f"{prefix}_overlay.jpg", vis)
                    print(f"[debug_stop] Saved {prefix}_frame_full.jpg, _frame_roi.jpg, _mask.jpg, _overlay.jpg")
                    print(f"  ROI: {STOP_SIGN_ROI}")
                    print(f"  area={stop_area}, circ={stop_details.get('circularity', 0):.2f}, "
                          f"raw_area={stop_details.get('raw_largest_area', 0)}, raw_circ={stop_details.get('raw_circularity', 0):.2f}, "
                          f"min_area={stop_detector.min_area}, min_circ={stop_detector.min_circularity}, "
                          f"centroid={stop_details.get('centroid')}, bbox={stop_details.get('bounding_box')}")
                
                # STOP THE ROBOT COMPLETELY
                bot.setVelocity(0, 0)
                
                # Wait for the required duration (rules say must come to complete stop)
                time.sleep(STOP_DURATION)
                
                # Mark that we've handled this stop sign
                stop_sign_handled = True
                last_stop_time = time.time()
                stop_history.clear()  # Reset rolling window after stopping
                
                print("Resuming...")
                continue  # Skip this iteration, resume driving on next loop
        
        # Reset the handled flag once we no longer see the stop sign
        # This allows us to detect and stop at the NEXT stop sign
        if not stop_detected_relaxed and not stop_detected_tight and stop_sign_handled:
            # Add a small delay before resetting to avoid flickering
            if current_time - last_stop_time > RESUME_COOLDOWN:
                stop_sign_handled = False
                stop_history.clear()  # Clean slate for next stop sign
        
        # ====================================================================
        # MOTOR CONTROL (with confidence-based speed adjustment)
        # ====================================================================
        Kd_base = 45  # Base wheel speeds at full confidence
        Ka_base = 45  # Turn rate at full confidence
        
        # Apply speed factor based on model confidence
        Kd = Kd_base * speed_factor
        Ka = Ka_base * speed_factor  # Also reduce turn rate when uncertain
        
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:
    pass
finally:
    bot.setVelocity(0, 0)
    if args.show:
        cv2.destroyAllWindows()
