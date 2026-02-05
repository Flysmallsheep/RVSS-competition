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
        min_area: int = 100,
        max_area: int = 2000,  # Blobs bigger than this are NOT stop signs (orange tiles!)
        min_circularity: float = 0.25,  # Stop signs are round (~0.5-0.8), orange edges are streaks (~0.05-0.2)
        # --- TIGHT thresholds (sharp frames, high confidence) ---
        # Red in HSV wraps around 0/180. We need two ranges:
        # Range 1: Hue 0-4 (tightened from 8 to exclude orange track which is ~6-15)
        # S_low increased to 130 to differentiate from duller orange
        lower_red1: tuple = (0, 130, 70),
        upper_red1: tuple = (4, 255, 255),
        # Range 2: Hue 176-180 (catch magenta-red)
        lower_red2: tuple = (176, 130, 70),
        upper_red2: tuple = (180, 255, 255),
        # --- RELAXED thresholds (motion-blurred frames) ---
        # Lower saturation (70 vs 130) catches washed-out red from blur.
        # Hue stays at 0-4 (same as tight) to avoid orange (hue ~8-15).
        # Use with multi-frame accumulation to guard against false positives.
        lower_red1_relaxed: tuple = (0, 70, 50),
        upper_red1_relaxed: tuple = (4, 255, 255),
        lower_red2_relaxed: tuple = (174, 70, 50),
        upper_red2_relaxed: tuple = (180, 255, 255),
        min_area_relaxed: int = None,  # defaults to min_area // 2
    ):
        """
        Initialize the stop sign detector.
        
        Two threshold tiers:
          - TIGHT (default): High saturation, narrow hue. Reliable on sharp frames.
            Use for single-frame detection or high-confidence triggers.
          - RELAXED: Low saturation, wider hue. Catches motion-blurred red.
            Use with multi-frame accumulation (rolling window) so that
            false positives are filtered out by requiring N/M hits.
        
        Args:
            min_area: Minimum blob area (in pixels) for tight detection.
            lower_red1, upper_red1: Tight HSV range for red hue near 0°
            lower_red2, upper_red2: Tight HSV range for red hue near 180°
            lower_red1_relaxed, upper_red1_relaxed: Relaxed HSV range near 0°
            lower_red2_relaxed, upper_red2_relaxed: Relaxed HSV range near 180°
            min_area_relaxed: Minimum blob area for relaxed detection (default: min_area // 2)
        
        HSV ranges explanation:
            H (Hue): 0-180 in OpenCV (0=red, 60=green, 120=blue)
            S (Saturation): 0-255 (0=gray, 255=pure color)
            V (Value): 0-255 (0=black, 255=bright)
            
            Red is tricky because it wraps around 0/180, so we use two ranges.
        """
        self.min_area = min_area
        self.max_area = max_area  # Upper bound: reject huge blobs (orange tiles)
        self.min_circularity = min_circularity  # Reject elongated streaks (orange tile edges)
        self.min_area_relaxed = min_area_relaxed if min_area_relaxed is not None else max(1, min_area // 2)
        
        # TIGHT HSV thresholds (sharp frames)
        self.lower_red1 = np.array(lower_red1, dtype=np.uint8)
        self.upper_red1 = np.array(upper_red1, dtype=np.uint8)
        self.lower_red2 = np.array(lower_red2, dtype=np.uint8)
        self.upper_red2 = np.array(upper_red2, dtype=np.uint8)
        
        # RELAXED HSV thresholds (motion-blurred frames)
        self.lower_red1_relaxed = np.array(lower_red1_relaxed, dtype=np.uint8)
        self.upper_red1_relaxed = np.array(upper_red1_relaxed, dtype=np.uint8)
        self.lower_red2_relaxed = np.array(lower_red2_relaxed, dtype=np.uint8)
        self.upper_red2_relaxed = np.array(upper_red2_relaxed, dtype=np.uint8)
    
    @staticmethod
    def _circularity(contour):
        """
        Compute circularity of a contour: 4π × area / perimeter².
        Circle = 1.0, elongated streak ≈ 0.05-0.2.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter * perimeter)
    
    def _make_red_mask(self, hsv, tight=True):
        """Create a red binary mask from an HSV image."""
        if tight:
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        else:
            mask1 = cv2.inRange(hsv, self.lower_red1_relaxed, self.upper_red1_relaxed)
            mask2 = cv2.inRange(hsv, self.lower_red2_relaxed, self.upper_red2_relaxed)
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        return red_mask
    
    def _find_best_contour(self, contours, min_area, max_area):
        """
        Find the best stop-sign candidate contour.
        Filters by area range AND circularity.
        Returns (contour, area) or (None, 0) if nothing passes.
        """
        best_contour = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            circ = self._circularity(c)
            if circ < self.min_circularity:
                continue
            # Pick the largest valid contour
            if area > best_area:
                best_area = area
                best_contour = c
        return best_contour, int(best_area)
    
    def detect(self, bgr_image: np.ndarray) -> tuple:
        """
        Detect if a stop sign is present and close enough (tight thresholds).
        
        Filters by:
          - Area range: min_area <= area <= max_area
          - Circularity: blob must be round-ish (rejects elongated orange streaks)
        
        Args:
            bgr_image: Input image in BGR format
        
        Returns:
            (stop_detected: bool, largest_area: int)
        """
        if bgr_image is None or bgr_image.size == 0:
            return False, 0
        
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        red_mask = self._make_red_mask(hsv, tight=True)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0
        
        best_contour, best_area = self._find_best_contour(contours, self.min_area, self.max_area)
        return best_contour is not None, best_area
    
    def detect_relaxed(self, bgr_image: np.ndarray) -> tuple:
        """
        Detect red using RELAXED thresholds (for motion-blurred frames).
        
        Lower saturation + circularity filter. Use with multi-frame accumulation.
        
        Args:
            bgr_image: Input image in BGR format
        
        Returns:
            (detected: bool, largest_area: int)
        """
        if bgr_image is None or bgr_image.size == 0:
            return False, 0
        
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        red_mask = self._make_red_mask(hsv, tight=False)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0
        
        best_contour, best_area = self._find_best_contour(contours, self.min_area_relaxed, self.max_area)
        return best_contour is not None, best_area
    
    def detect_both(self, bgr_image: np.ndarray) -> dict:
        """
        Run both tight and relaxed detection in one call.
        
        Returns:
            dict with keys:
                - tight_detected: bool (high confidence, sharp frame)
                - tight_area: int
                - relaxed_detected: bool (catches blur, use with rolling window)
                - relaxed_area: int
        """
        tight_detected, tight_area = self.detect(bgr_image)
        relaxed_detected, relaxed_area = self.detect_relaxed(bgr_image)
        return {
            'tight_detected': tight_detected,
            'tight_area': tight_area,
            'relaxed_detected': relaxed_detected,
            'relaxed_area': relaxed_area,
        }
    
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
        red_mask = self._make_red_mask(hsv, tight=True)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'detected': False,
                'largest_area': 0,
                'circularity': 0.0,
                'centroid': None,
                'bounding_box': None,
                'mask': red_mask,
                'all_areas': []
            }
        
        # Get all areas and circularities for debugging
        all_areas = [(cv2.contourArea(c), self._circularity(c)) for c in contours]
        
        # Find best valid contour (area + circularity filter)
        best_contour, best_area = self._find_best_contour(contours, self.min_area, self.max_area)
        
        # Also report the raw largest contour for debugging
        raw_largest = max(contours, key=cv2.contourArea)
        raw_largest_area = cv2.contourArea(raw_largest)
        raw_circ = self._circularity(raw_largest)
        
        # Use best valid contour for centroid/bbox, fall back to raw largest for debugging
        report_contour = best_contour if best_contour is not None else raw_largest
        report_area = best_area if best_contour is not None else int(raw_largest_area)
        
        # Compute centroid using moments
        M = cv2.moments(report_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = (cx, cy)
        else:
            centroid = None
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(report_contour)
        
        return {
            'detected': best_contour is not None,
            'largest_area': report_area,
            'circularity': self._circularity(report_contour),
            'raw_largest_area': int(raw_largest_area),
            'raw_circularity': raw_circ,
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
            stop_detected = self.min_area <= largest_area <= self.max_area
            
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
