import cv2
import mediapipe as mp
import numpy as np

class PeopleDetection:
    def __init__(self, cap, hands, mp_hands, mp_drawing):
        self.cap = cap
        self.hands = hands
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        
        # Use only HOG detector to avoid MediaPipe face detection issues
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.hog_available = True
            print("HOG people detector initialized successfully")
        except Exception as e:
            print(f"HOG detector failed: {e}")
            self.hog_available = False
        
        # Backup: Use background subtraction for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        self.people_count = 0
        self.detected_boxes = []
        self.return_frame_count = 0
        self.FRAMES_TO_CONFIRM = 15
        
        # Colors for different people
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

    def _detect_people_hog(self, frame):
        """Detect people using HOG descriptor"""
        if not self.hog_available:
            return []
        
        try:
            # Resize for better performance
            height, width = frame.shape[:2]
            scale = 0.5
            small_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
            # Detect people
            boxes, weights = self.hog.detectMultiScale(
                small_frame,
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.05,
                finalThreshold=1.0
            )
            
            detected_boxes = []
            
            if len(boxes) > 0:
                # Scale boxes back to original size
                for (x, y, w, h) in boxes:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)
                    
                    # Add some padding
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(width - x, w + 2 * padding)
                    h = min(height - y, h + 2 * padding)
                    
                    detected_boxes.append((x, y, x + w, y + h))
            
            return detected_boxes
        except Exception as e:
            print(f"HOG detection error: {e}")
            return []

    def _detect_people_motion(self, frame):
        """Detect people using motion detection as backup"""
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_boxes = []
            min_area = 5000  # Minimum area for a person
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (people are taller than wide)
                    aspect_ratio = h / w
                    if 1.2 < aspect_ratio < 4.0:  # Reasonable aspect ratio for a person
                        detected_boxes.append((x, y, x + w, y + h))
            
            return detected_boxes
        except Exception as e:
            print(f"Motion detection error: {e}")
            return []

    def _filter_boxes(self, boxes):
        """Remove overlapping and invalid boxes"""
        if len(boxes) <= 1:
            return boxes
        
        filtered = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Check if box is valid
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Check for significant overlap with existing boxes
            is_duplicate = False
            for existing_box in filtered:
                x3, y3, x4, y4 = existing_box
                
                # Calculate intersection
                ix1 = max(x1, x3)
                iy1 = max(y1, y3)
                ix2 = min(x2, x4)
                iy2 = min(y2, y4)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (x4 - x3) * (y4 - y3)
                    
                    # If overlap is more than 50% of smaller box, consider it duplicate
                    overlap_ratio = intersection / min(area1, area2)
                    if overlap_ratio > 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(box)
        
        return filtered

    def _detect_people(self, frame):
        """Main detection function"""
        all_boxes = []
        
        # Method 1: HOG detection (if available)
        if self.hog_available:
            hog_boxes = self._detect_people_hog(frame)
            all_boxes.extend(hog_boxes)
        
        # Method 2: Motion detection (as backup)
        motion_boxes = self._detect_people_motion(frame)
        all_boxes.extend(motion_boxes)
        
        # Filter and return
        return self._filter_boxes(all_boxes)

    def _draw_detections(self, frame, boxes):
        """Draw bounding boxes around detected people"""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw person label
            label = f"Person {i + 1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Background for label
            cv2.rectangle(frame, 
                         (x1, y1 - label_size[1] - 15),
                         (x1 + label_size[0] + 10, y1),
                         color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, color, -1)

    def _draw_info_panel(self, frame):
        """Draw information panel"""
        # Background panel
        cv2.rectangle(frame, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 140), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "People Detection", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # People count
        count_text = f"People detected: {self.people_count}"
        cv2.putText(frame, count_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection method
        method = "HOG + Motion" if self.hog_available else "Motion Only"
        cv2.putText(frame, f"Method: {method}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Status
        status = "Detecting..." if self.people_count > 0 else "Waiting..."
        cv2.putText(frame, f"Status: {status}", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.people_count > 0 else (255, 255, 0), 2)

    def _draw_instructions(self, frame):
        """Draw instructions"""
        instructions = [
            "Move around to be detected by motion sensor",
            "Stand still for HOG detection to work",
            "Multiple people supported",
            "Press 'r' to reset background | 'q' to quit"
        ]
        
        y_start = frame.shape[0] - 110
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, y_start + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _draw_return_button(self, frame):
        """Draw return to menu button"""
        cv2.rectangle(frame, (50, 650), (250, 700), (0, 0, 255), -1)
        cv2.putText(frame, "Voltar menu", (70, 685),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _check_return_button(self, x, y):
        """Check if return button is pressed"""
        return 50 < x < 250 and 650 < y < 700

    def run(self):
        """Main detection loop"""
        print("People Detection started...")
        print(f"HOG detection: {'Available' if self.hog_available else 'Not available'}")
        print("Motion detection: Available")
        print("Move around to be detected!")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Detect people
                detected_boxes = self._detect_people(frame)
                self.people_count = len(detected_boxes)
                
                # Draw detections
                if detected_boxes:
                    self._draw_detections(frame, detected_boxes)
                
                # Draw UI elements
                self._draw_info_panel(frame)
                self._draw_instructions(frame)
                self._draw_return_button(frame)
                
                # Check for hand gestures to return to menu
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_results = self.hands.process(rgb_frame)
                    
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # Draw hand landmarks
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                            )
                            
                            # Check return button
                            index_tip = hand_landmarks.landmark[8]
                            x = int(index_tip.x * frame.shape[1])
                            y = int(index_tip.y * frame.shape[0])
                            
                            if self._check_return_button(x, y):
                                self.return_frame_count += 1
                                # Draw progress circle
                                progress = self.return_frame_count / self.FRAMES_TO_CONFIRM
                                radius = int(progress * 20)
                                cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
                                
                                if self.return_frame_count >= self.FRAMES_TO_CONFIRM:
                                    cv2.destroyWindow('People Detection')
                                    return True  # Return to menu
                            else:
                                self.return_frame_count = 0
                except Exception as e:
                    print(f"Hand processing error: {e}")
                
                cv2.imshow('People Detection', frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Resetting background model...")
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        except Exception as e:
            print(f"Error in detection loop: {e}")
        
        finally:
            cv2.destroyWindow('People Detection')
        
        return False  # Exit application
