import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

class PeopleDetection:
    def __init__(self, cap, hands, mp_hands, mp_drawing):
        self.cap = cap
        self.hands = hands
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        
        # Initialize YOLO model (will download automatically on first run)
        try:
            self.model = YOLO('yolov8n.pt')  # nano version for speed
            self.yolo_available = True
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"YOLO failed to load: {e}")
            self.yolo_available = False
            # Fallback to simple background subtraction
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        self.people_count = 0
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
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]

    def _detect_people_yolo(self, frame):
        """Detect people using YOLO - much more accurate"""
        if not self.yolo_available:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            detected_boxes = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class (0 = person in COCO dataset)
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Only detect persons with good confidence
                        if class_id == 0 and confidence > 0.5:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detected_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            return detected_boxes
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def _detect_people_simple_motion(self, frame):
        """Simple motion detection fallback"""
        if self.yolo_available:
            return []  # Only use if YOLO is not available
        
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_boxes = []
            min_area = 3000  # Minimum area for a person
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (people are taller than wide)
                    aspect_ratio = h / w if w > 0 else 0
                    if 1.2 < aspect_ratio < 4.0:
                        detected_boxes.append((x, y, x + w, y + h))
            
            return detected_boxes
        except Exception as e:
            print(f"Motion detection error: {e}")
            return []

    def _detect_people(self, frame):
        """Main detection function"""
        if self.yolo_available:
            return self._detect_people_yolo(frame)
        else:
            return self._detect_people_simple_motion(frame)

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

    def _draw_info_panel(self, frame):
        """Draw information panel"""
        # Background panel
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 120), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "People Detection", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # People count
        count_text = f"People: {self.people_count}"
        cv2.putText(frame, count_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection method
        method = "YOLO" if self.yolo_available else "Motion"
        cv2.putText(frame, f"Method: {method}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _draw_instructions(self, frame):
        """Draw instructions"""
        if self.yolo_available:
            instructions = [
                "YOLO detection active - just stand in view",
                "Multiple people supported automatically",
                "Point at 'Voltar menu' to return"
            ]
        else:
            instructions = [
                "Motion detection - move to be detected",
                "Press 'r' to reset background",
                "Point at 'Voltar menu' to return"
            ]
        
        y_start = frame.shape[0] - 90
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
        if self.yolo_available:
            print("Using YOLO detection - high accuracy, multiple people supported")
        else:
            print("Using motion detection - move around to be detected")
        
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
                elif key == ord('r') and not self.yolo_available:
                    print("Resetting background model...")
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        except Exception as e:
            print(f"Error in detection loop: {e}")
        
        finally:
            cv2.destroyWindow('People Detection')
        
        return False  # Exit application
