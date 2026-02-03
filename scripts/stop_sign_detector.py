#!/usr/bin/env python3
"""
Stop Sign Detector using Color Thresholding + Blob Detection

This module detects red stop signs using classical computer vision:
1. Convert BGR image to HSV color space
2. Threshold for red color (stop sign is red)
3. Find contours/blobs in the thresholded image
4. Filter by area to determine if the sign is "close enough" to stop

No training required! Just tune the thresholds if needed.

Notes from instructions/StopSignDetection.md:
- cv2 loads images in BGR; machinevision-toolbox expects RGB
- blobs() can throw an error if no blobs found → use try/except
- Stop sign can appear on urban or rural tiles → thresholds must be robust
"""

import cv2
import numpy as np

# Optional: machinevision-toolbox (as recommended in instructions)
# We'll use OpenCV by default since it's more portable, but the toolbox
# can be used as an alternative for blob features (centroid, circularity, etc.)
try:
    from machinevisiontoolbox import Image as MVTImage
    MVT_AVAILABLE = True
except ImportError:
    MVT_AVAILABLE = False


class StopSignDetector:
    """
    Detects red stop signs using HSV color thresholding and blob detection.
    
    The detector looks for red-colored regions in the image. When a red blob
    exceeds the area threshold, it means the stop sign is close enough and
    the robot should stop.
    
    Attributes:
        min_area: Minimum blob area (pixels) to trigger a stop.
                  Larger = must be closer to trigger. Tune based on your camera.
        lower_red1, upper_red1: HSV range for red (hue near 0)
        lower_red2, upper_red2: HSV range for red (hue near 180)
    """
    
    def __init__(
        self,
        min_area: int = 500,
        # Red in HSV wraps around 0/180. We need two ranges:
        # Range 1: Hue 0-10 (red-orange side)
        lower_red1: tuple = (0, 100, 100),
        upper_red1: tuple = (10, 255, 255),
        # Range 2: Hue 160-180 (red-magenta side)
        lower_red2: tuple = (160, 100, 100),
        upper_red2: tuple = (180, 255, 255),
    ):
        """
        Initialize the stop sign detector.
        
        Args:
            min_area: Minimum blob area (in pixels) to consider the stop sign
                      "close enough" to trigger a stop. 
                      - Smaller value = detect from farther away
                      - Larger value = must be very close
                      Default 500 is a starting point; tune based on testing.
            lower_red1, upper_red1: HSV range for red hue near 0°
            lower_red2, upper_red2: HSV range for red hue near 180°
        
        HSV ranges explanation:
            H (Hue): 0-180 in OpenCV (0=red, 60=green, 120=blue)
            S (Saturation): 0-255 (0=gray, 255=pure color)
            V (Value): 0-255 (0=black, 255=bright)
            
            Red is tricky because it wraps around 0/180, so we use two ranges.
        """
        self.min_area = min_area
        
        # HSV thresholds for red detection
        self.lower_red1 = np.array(lower_red1, dtype=np.uint8)
        self.upper_red1 = np.array(upper_red1, dtype=np.uint8)
        self.lower_red2 = np.array(lower_red2, dtype=np.uint8)
        self.upper_red2 = np.array(upper_red2, dtype=np.uint8)
    
    def detect(self, bgr_image: np.ndarray) -> tuple:
        """
        Detect if a stop sign is present and close enough.
        
        Args:
            bgr_image: Input image in BGR format (as returned by cv2.imread or bot.getImage())
        
        Returns:
            (stop_detected: bool, largest_area: int)
            - stop_detected: True if a red blob with area >= min_area is found
            - largest_area: Area of the largest red blob (0 if none found)
        
        Example:
            detector = StopSignDetector(min_area=500)
            stop_detected, area = detector.detect(image)
            if stop_detected:
                print(f"Stop sign detected! Area: {area}")
        """
        if bgr_image is None or bgr_image.size == 0:
            return False, 0
        
        # Step 1: Convert BGR to HSV
        # HSV is better for color-based segmentation because it separates
        # color (Hue) from brightness (Value), making it more robust to lighting.
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Step 2: Create masks for red color
        # Red in HSV wraps around 0 and 180, so we need two masks.
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        
        # Combine both masks (logical OR)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Step 3: Clean up the mask with morphological operations
        # This removes small noise and fills small holes.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        
        # Step 4: Find contours (blobs) in the mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0
        
        # Step 5: Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        # Step 6: Check if the area exceeds our threshold
        # Larger area = closer to the sign
        stop_detected = largest_area >= self.min_area
        
        return stop_detected, int(largest_area)
    
    def detect_with_details(self, bgr_image: np.ndarray) -> dict:
        """
        Detect stop sign and return detailed information (for debugging/tuning).
        
        Args:
            bgr_image: Input image in BGR format
        
        Returns:
            dict with keys:
                - detected: bool
                - largest_area: int
                - centroid: (x, y) tuple or None
                - bounding_box: (x, y, w, h) tuple or None
                - mask: binary mask image (for visualization)
                - all_areas: list of all blob areas found
        """
        if bgr_image is None or bgr_image.size == 0:
            return {
                'detected': False,
                'largest_area': 0,
                'centroid': None,
                'bounding_box': None,
                'mask': None,
                'all_areas': []
            }
        
        # Same detection pipeline
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'detected': False,
                'largest_area': 0,
                'centroid': None,
                'bounding_box': None,
                'mask': red_mask,
                'all_areas': []
            }
        
        # Get all areas
        all_areas = [cv2.contourArea(c) for c in contours]
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        # Compute centroid using moments
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = (cx, cy)
        else:
            centroid = None
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return {
            'detected': largest_area >= self.min_area,
            'largest_area': int(largest_area),
            'centroid': centroid,
            'bounding_box': (x, y, w, h),
            'mask': red_mask,
            'all_areas': all_areas
        }
    
    def detect_with_mvt(self, bgr_image: np.ndarray) -> tuple:
        """
        Alternative detection using machinevision-toolbox (as suggested in instructions).
        
        This uses Peter Corke's machinevision-toolbox for blob detection.
        Provides additional features like circularity, perimeter, etc.
        
        Note: machinevision-toolbox expects RGB, not BGR!
        
        Args:
            bgr_image: Input image in BGR format
        
        Returns:
            (stop_detected: bool, largest_area: int)
        """
        if not MVT_AVAILABLE:
            print("Warning: machinevision-toolbox not available, falling back to OpenCV")
            return self.detect(bgr_image)
        
        if bgr_image is None or bgr_image.size == 0:
            return False, 0
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Create red mask
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Use machinevision-toolbox for blob detection
        # Note: MVT expects the image as-is (binary mask is fine)
        try:
            mvt_image = MVTImage(red_mask)
            blobs = mvt_image.blobs()
            
            if blobs is None or len(blobs) == 0:
                return False, 0
            
            # Find largest blob by area
            largest_area = max(b.area for b in blobs)
            stop_detected = largest_area >= self.min_area
            
            return stop_detected, int(largest_area)
            
        except Exception as e:
            # blobs() can throw an error if no blobs found
            # This is mentioned in instructions/StopSignDetection.md
            return False, 0


def tune_thresholds(bgr_image: np.ndarray):
    """
    Interactive tool to tune HSV thresholds for red detection.
    
    Opens a window with trackbars to adjust HSV ranges in real-time.
    Use this to find the best thresholds for your lighting conditions.
    
    Args:
        bgr_image: A sample image containing the stop sign
    
    Usage:
        # Capture an image with stop sign visible
        img = bot.getImage()
        tune_thresholds(img)
    """
    def nothing(x):
        pass
    
    cv2.namedWindow('Threshold Tuning')
    
    # Create trackbars for red range 1
    cv2.createTrackbar('H1_low', 'Threshold Tuning', 0, 180, nothing)
    cv2.createTrackbar('H1_high', 'Threshold Tuning', 10, 180, nothing)
    cv2.createTrackbar('S_low', 'Threshold Tuning', 100, 255, nothing)
    cv2.createTrackbar('V_low', 'Threshold Tuning', 100, 255, nothing)
    
    # Create trackbars for red range 2
    cv2.createTrackbar('H2_low', 'Threshold Tuning', 160, 180, nothing)
    cv2.createTrackbar('H2_high', 'Threshold Tuning', 180, 180, nothing)
    
    print("Adjust trackbars to tune thresholds. Press 'q' to quit.")
    print("White areas in the mask = detected red regions")
    
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    
    while True:
        h1_low = cv2.getTrackbarPos('H1_low', 'Threshold Tuning')
        h1_high = cv2.getTrackbarPos('H1_high', 'Threshold Tuning')
        s_low = cv2.getTrackbarPos('S_low', 'Threshold Tuning')
        v_low = cv2.getTrackbarPos('V_low', 'Threshold Tuning')
        h2_low = cv2.getTrackbarPos('H2_low', 'Threshold Tuning')
        h2_high = cv2.getTrackbarPos('H2_high', 'Threshold Tuning')
        
        lower1 = np.array([h1_low, s_low, v_low])
        upper1 = np.array([h1_high, 255, 255])
        lower2 = np.array([h2_low, s_low, v_low])
        upper2 = np.array([h2_high, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Show original and mask side by side
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([bgr_image, mask_colored])
        cv2.imshow('Threshold Tuning', combined)
        
        # Print current values
        print(f"\rH1: {h1_low}-{h1_high}, H2: {h2_low}-{h2_high}, S: {s_low}+, V: {v_low}+", end='')
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    print("\n\nFinal thresholds:")
    print(f"lower_red1 = ({h1_low}, {s_low}, {v_low})")
    print(f"upper_red1 = ({h1_high}, 255, 255)")
    print(f"lower_red2 = ({h2_low}, {s_low}, {v_low})")
    print(f"upper_red2 = ({h2_high}, 255, 255)")
    
    cv2.destroyAllWindows()


# ============================================================================
# Quick test / demo
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test stop sign detector')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--tune', action='store_true', help='Open threshold tuning tool')
    parser.add_argument('--min_area', type=int, default=500, help='Minimum blob area')
    args = parser.parse_args()
    
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}")
            exit(1)
        
        if args.tune:
            tune_thresholds(img)
        else:
            detector = StopSignDetector(min_area=args.min_area)
            result = detector.detect_with_details(img)
            
            print(f"Stop sign detected: {result['detected']}")
            print(f"Largest area: {result['largest_area']}")
            print(f"Centroid: {result['centroid']}")
            print(f"Bounding box: {result['bounding_box']}")
            print(f"All areas: {result['all_areas']}")
            
            # Visualize
            if result['mask'] is not None:
                cv2.imshow('Original', img)
                cv2.imshow('Red Mask', result['mask'])
                
                # Draw bounding box on original
                if result['bounding_box']:
                    x, y, w, h = result['bounding_box']
                    vis = img.copy()
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    if result['centroid']:
                        cv2.circle(vis, result['centroid'], 5, (0, 0, 255), -1)
                    cv2.imshow('Detection', vis)
                
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        print("Usage:")
        print("  Test on image:  python stop_sign_detector.py --image path/to/image.jpg")
        print("  Tune thresholds: python stop_sign_detector.py --image path/to/image.jpg --tune")
        print("  With custom area: python stop_sign_detector.py --image path/to/image.jpg --min_area 1000")
