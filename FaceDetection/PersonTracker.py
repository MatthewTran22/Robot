"""
PersonTracker - Continuous object tracking using OpenCV
Tracks a person across frames even when face is not visible
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class PersonTracker:
    """
    Track a person continuously using OpenCV object tracker, even when face is not visible
    """
    def __init__(self, person_name: str, initial_box: Tuple[int, int, int, int], 
                 frame: np.ndarray, confidence: float, enable_body_tracking: bool = True):
        """
        Initialize person tracker
        
        Args:
            person_name: Name of the tracked person
            initial_box: Initial bounding box (x, y, w, h)
            frame: Current video frame
            confidence: Recognition confidence score
            enable_body_tracking: Expand box to cover full body
        """
        self.person_name = person_name
        self.confidence = confidence
        self.enable_body_tracking = enable_body_tracking
        self.current_box = initial_box
        
        # Initialize OpenCV object tracker (KCF is fast and reliable)
        self.tracker = cv2.TrackerKCF_create()
        
        # Expand initial box for full body tracking
        if enable_body_tracking:
            x, y, w, h = initial_box
            # Expand to approximate body size
            body_w = int(w * 2.5)
            body_h = int(h * 5)
            body_x = max(0, x - int((body_w - w) / 2))
            body_y = max(0, y - int(h * 0.5))
            
            # Ensure within frame
            h_frame, w_frame = frame.shape[:2]
            body_w = min(body_w, w_frame - body_x)
            body_h = min(body_h, h_frame - body_y)
            
            self.current_box = (body_x, body_y, body_w, body_h)
        
        # Initialize tracker with the box
        x, y, w, h = self.current_box
        self.tracker.init(frame, (x, y, w, h))
        
        # Face re-identification tracking
        self.frames_since_face_seen = 0
        self.max_frames_without_face = 180  # Re-identify every ~6 seconds (at 30fps)
        self.last_face_box = initial_box
        
    def update_tracking(self, frame: np.ndarray) -> bool:
        """
        Update object tracker position
        
        Args:
            frame: Current video frame
            
        Returns:
            True if tracking successful, False if lost
        """
        success, box = self.tracker.update(frame)
        
        if success:
            self.current_box = tuple(map(int, box))
            self.frames_since_face_seen += 1
            return True
        else:
            # Tracking lost
            return False
    
    def update_with_face(self, face_box: Tuple[int, int, int, int], 
                         frame: np.ndarray, confidence: float):
        """
        Re-initialize tracker when face is detected again
        
        Args:
            face_box: New face bounding box (x, y, w, h)
            frame: Current video frame
            confidence: Recognition confidence score
        """
        self.confidence = confidence
        self.frames_since_face_seen = 0
        self.last_face_box = face_box
        
        # Expand to body box if enabled
        if self.enable_body_tracking:
            x, y, w, h = face_box
            body_w = int(w * 2.5)
            body_h = int(h * 5)
            body_x = max(0, x - int((body_w - w) / 2))
            body_y = max(0, y - int(h * 0.5))
            
            h_frame, w_frame = frame.shape[:2]
            body_w = min(body_w, w_frame - body_x)
            body_h = min(body_h, h_frame - body_y)
            
            new_box = (body_x, body_y, body_w, body_h)
        else:
            new_box = face_box
        
        # Reinitialize tracker with updated position
        self.current_box = new_box
        self.tracker = cv2.TrackerKCF_create()
        x, y, w, h = new_box
        self.tracker.init(frame, (x, y, w, h))
    
    def get_box(self) -> Tuple[int, int, int, int]:
        """
        Get current tracking box
        
        Returns:
            Current bounding box (x, y, w, h)
        """
        return self.current_box
    
    def needs_face_reidentification(self) -> bool:
        """
        Check if we need to re-identify face
        
        Returns:
            True if too long without face detection
        """
        return self.frames_since_face_seen > self.max_frames_without_face
    
    def overlaps_with_face(self, face_box: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
        """
        Check if a detected face overlaps with current tracking box
        
        Args:
            face_box: Face bounding box (x, y, w, h)
            threshold: Minimum IoU overlap threshold
            
        Returns:
            True if face is within tracked region
        """
        x1, y1, w1, h1 = self.current_box
        x2, y2, w2, h2 = face_box
        
        # Calculate intersection over union (IoU)
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return False
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > threshold

