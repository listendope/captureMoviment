import cv2
import mediapipe as mp
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")

class PeopleDetection:
    def __init__(self, cap, hands, mp_hands, mp_drawing):
        self.cap = cap
        self.hands = hands
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        
        # Default screen size (will be updated by main.py)
        self.screen_width = 1280
        self.screen_height = 720
        
        # Initialize YOLO model if available
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n.pt')
                self.yolo_available = True
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f"YOLO failed to load: {e}")
                self.yolo_available = False
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        else:
            self.yolo_available = False
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        self.people_count = 0
        self.return_frame_count = 0
        self.FRAMES_TO_CONFIRM = 15
        
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]

    def set_screen_size(self, width, height):
        """Set screen dimensions for proper window sizing"""
        self.screen_width = width
        self.screen_height = height

    def _setup_window(self, window_name):
        """Setup window to fit screen properly"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.screen_width, self.screen_height)
        cv2.moveWindow(window_name, 0, 0)

    def _resize_frame(self, frame):
        """Resize frame to fit screen while maintaining aspect ratio"""
        frame_height, frame_width = frame.shape[:2]
        
        scale_x = self.screen_width / frame_width
        scale_y = self.screen_height / frame_height
        scale = min(scale_x, scale_y)
        
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        final_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        y_offset = (self.screen_height - new_height) // 2
        x_offset = (self.screen_width - new_width) // 2
        
        final_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        
        return final_frame, scale, x_offset, y_offset

    def _adjust_coordinates(self, x, y, scale, x_offset, y_offset):
        """Adjust coordinates based on frame scaling"""
        adjusted_x = int((x * scale) + x_offset)
        adjusted_y = int((y * scale) + y_offset)
        return adjusted_x, adjusted_y

    def _detect_people_yolo(self, frame):
        """Detect people using YOLO"""
        if not self.yolo_available:
            return []
        
        try:
            results = self.model(frame, verbose=False)
            detected_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == 0 and confidence > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detected_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            return detected_boxes
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def _detect_people_simple_motion(self, frame):
        """Simple motion detection fallback"""
        if self.yolo_available:
            return []
        
        try:
            fg_mask = self.bg_subtractor.apply(frame)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_boxes = []
            min_area = 3000
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
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

    def _draw_detections(self, frame, boxes, scale, x_offset, y_offset):
        """Draw bounding boxes around detected people"""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Adjust coordinates for display
            adj_x1, adj_y1 = self._adjust_coordinates(x1, y1, scale, x_offset, y_offset)
            adj_x2, adj_y2 = self._adjust_coordinates(x2, y2, scale, x_offset, y_offset)
            
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), color, 3)
            
            # Draw person label
            label = f"Person {i + 1}"
            font_scale = max(0.6, self.screen_width / 2000)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            
            # Background for label
            cv2.rectangle(frame, 
                         (adj_x1, adj_y1 - label_size[1] - 15),
                         (adj_x1 + label_size[0] + 10, adj_y1),
                         color, -1)
            
            # Label text
            cv2.putText(frame, label, (adj_x1 + 5, adj_y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    def _draw_info_panel(self, frame):
        """Draw information panel"""
        panel_width = max(350, self.screen_width // 4)
        panel_height = max(120, self.screen_height // 6)
        
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
        
        font_scale = max(0.8, self.screen_width / 1600)
        
        # Title
        cv2.putText(frame, "People Detection", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        
        # People count
        count_text = f"People: {self.people_count}"
        cv2.putText(frame, count_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), 2)
        
        # Detection method
        method = "YOLO" if self.yolo_available else "Motion"
        cv2.putText(frame, f"Method: {method}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 255, 255), 2)

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
        
        font_scale = max(0.5, self.screen_width / 2500)
        y_start = self.screen_height - 90
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, y_start + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    def _draw_return_button(self, frame):
        """Draw return to menu button"""
        button_width = max(200, self.screen_width // 8)
        button_height = max(50, self.screen_height // 20)
        button_x = max(50, self.screen_width // 25)
        button_y = self.screen_height - button_height - max(50, self.screen_height // 15)
        
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), (0, 0, 255), -1)
        
        font_scale = max(0.6, self.screen_width / 2000)
        cv2.putText(frame, "Voltar menu", (button_x + 20, button_y + button_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    def _check_return_button(self, x, y):
        """Check if return button is pressed"""
        button_width = max(200, self.screen_width // 8)
        button_height = max(50, self.screen_height // 20)
        button_x = max(50, self.screen_width // 25)
        button_y = self.screen_height - button_height - max(50, self.screen_height // 15)
        
        return (button_x < x < button_x + button_width and 
                button_y < y < button_y + button_height)

    def run(self):
        """Main detection loop"""
        print("People Detection started...")
        if self.yolo_available:
            print("Using YOLO detection - high accuracy, multiple people supported")
        else:
            print("Using motion detection - move around to be detected")
        
        # Setup window
        self._setup_window('People Detection')
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Resize frame to fit screen
                display_frame, scale, x_offset, y_offset = self._resize_frame(frame)
                
                # Detect people on original frame
                detected_boxes = self._detect_people(frame)
                self.people_count = len(detected_boxes)
                
                # Draw detections on display frame
                if detected_boxes:
                    self._draw_detections(display_frame, detected_boxes, scale, x_offset, y_offset)
                
                # Draw UI elements
                self._draw_info_panel(display_frame)
                self._draw_instructions(display_frame)
                self._draw_return_button(display_frame)
                
                # Check for hand gestures to return to menu
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_results = self.hands.process(rgb_frame)
                    
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # Draw hand landmarks on display frame
                            for landmark in hand_landmarks.landmark:
                                orig_x = int(landmark.x * frame.shape[1])
                                orig_y = int(landmark.y * frame.shape[0])
                                adj_x, adj_y = self._adjust_coordinates(orig_x, orig_y, scale, x_offset, y_offset)
                                cv2.circle(display_frame, (adj_x, adj_y), 3, (0, 255, 0), -1)
                            
                            # Check return button
                            index_tip = hand_landmarks.landmark[8]
                            orig_x = int(index_tip.x * frame.shape[1])
                            orig_y = int(index_tip.y * frame.shape[0])
                            x, y = self._adjust_coordinates(orig_x, orig_y, scale, x_offset, y_offset)
                            
                            if self._check_return_button(x, y):
                                self.return_frame_count += 1
                                # Draw progress circle
                                progress = self.return_frame_count / self.FRAMES_TO_CONFIRM
                                radius = int(progress * 20)
                                cv2.circle(display_frame, (x, y), radius, (0, 255, 255), 2)
                                
                                if self.return_frame_count >= self.FRAMES_TO_CONFIRM:
                                    cv2.destroyWindow('People Detection')
                                    return True  # Return to menu
                            else:
                                self.return_frame_count = 0
                except Exception as e:
                    print(f"Hand processing error: {e}")
                
                cv2.imshow('People Detection', display_frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and not self.yolo_available:
                    print("Resetting background subtractor...")
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
                
        except KeyboardInterrupt:
            print("Detection interrupted by user")
        except Exception as e:
            print(f"Error in people detection: {e}")
        finally:
            cv2.destroyWindow('People Detection')
            print("People Detection ended")
        
        return False

    def cleanup(self):
        """Clean up resources"""
        try:
            cv2.destroyAllWindows()
        except:
            pass

    def get_detection_stats(self):
        """Get current detection statistics"""
        return {
            'people_count': self.people_count,
            'detection_method': 'YOLO' if self.yolo_available else 'Motion',
            'yolo_available': self.yolo_available
        }

    def reset_background(self):
        """Reset background subtractor (for motion detection)"""
        if not self.yolo_available:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            print("Background subtractor reset")

    def set_detection_sensitivity(self, sensitivity='medium'):
        """Set detection sensitivity for motion detection"""
        if not self.yolo_available:
            if sensitivity == 'low':
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=True, varThreshold=50, history=200)
            elif sensitivity == 'high':
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=True, varThreshold=16, history=500)
            else:  # medium
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=True, varThreshold=25, history=300)
            
            print(f"Detection sensitivity set to: {sensitivity}")

    def toggle_detection_method(self):
        """Toggle between available detection methods"""
        if YOLO_AVAILABLE and not self.yolo_available:
            try:
                self.model = YOLO('yolov8n.pt')
                self.yolo_available = True
                print("Switched to YOLO detection")
            except Exception as e:
                print(f"Failed to switch to YOLO: {e}")
        elif self.yolo_available:
            self.yolo_available = False
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            print("Switched to motion detection")

    def get_people_positions(self, frame):
        """Get current positions of detected people"""
        detected_boxes = self._detect_people(frame)
        positions = []
        
        for i, (x1, y1, x2, y2) in enumerate(detected_boxes):
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            positions.append({
                'id': i + 1,
                'center': (center_x, center_y),
                'bbox': (x1, y1, x2, y2),
                'size': (width, height)
            })
        
        return positions

    def is_person_in_area(self, frame, area_coords):
        """Check if any person is in a specific area"""
        detected_boxes = self._detect_people(frame)
        area_x1, area_y1, area_x2, area_y2 = area_coords
        
        for x1, y1, x2, y2 in detected_boxes:
            # Check if person's center is in the area
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if (area_x1 <= center_x <= area_x2 and 
                area_y1 <= center_y <= area_y2):
                return True
        
        return False

    def get_largest_person(self, frame):
        """Get the bounding box of the largest detected person"""
        detected_boxes = self._detect_people(frame)
        
        if not detected_boxes:
            return None
        
        largest_box = None
        largest_area = 0
        
        for box in detected_boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            
            if area > largest_area:
                largest_area = area
                largest_box = box
        
        return largest_box

    def draw_detection_zones(self, frame, zones):
        """Draw detection zones on the frame"""
        for i, zone in enumerate(zones):
            x1, y1, x2, y2 = zone['coords']
            color = zone.get('color', (255, 255, 0))
            label = zone.get('label', f'Zone {i+1}')
            
            # Draw zone rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw zone label
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Check if person is in zone
            if self.is_person_in_area(frame, (x1, y1, x2, y2)):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, "OCCUPIED", (x1, y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def save_detection_frame(self, frame, filename=None):
        """Save current frame with detections"""
        if filename is None:
            import datetime
            timestamp = datetime.datetime.now().str
