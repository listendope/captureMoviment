import cv2
import mediapipe as mp
import numpy as np
import random

class VirtualPaint:
    def __init__(self, cap, hands, mp_hands, mp_drawing):
        self.cap = cap
        self.hands = hands
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        
        # Default screen size (will be updated by main.py)
        self.screen_width = 1280
        self.screen_height = 720
        
        _, frame = self.cap.read()
        self.canvas = np.zeros_like(frame)
        
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'white': (255, 255, 255),
            'orange': (0, 165, 255),
            'pink': (147, 20, 255),
            'cyan': (255, 255, 0)
        }
        
        self.color_boxes = {}
        self.selected_color = self.colors['red']
        self.brush_thickness = 5
        self.eraser_thickness = 20
        self.prev_point = None
        self.frame_count = 0

    def set_screen_size(self, width, height):
        """Set screen dimensions for proper window sizing"""
        self.screen_width = width
        self.screen_height = height
        self.color_boxes = self._create_color_boxes()

    def _setup_window(self, window_name):
        """Setup window to fit screen properly"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.screen_width, self.screen_height)
        cv2.moveWindow(window_name, 0, 0)

    def _resize_frame(self, frame):
        """Resize frame to fit screen while maintaining aspect ratio"""
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate scaling factor
        scale_x = self.screen_width / frame_width
        scale_y = self.screen_height / frame_height
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Create black background with screen size
        final_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Center the resized frame
        y_offset = (self.screen_height - new_height) // 2
        x_offset = (self.screen_width - new_width) // 2
        
        final_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        
        return final_frame, scale, x_offset, y_offset

    def _adjust_coordinates(self, x, y, scale, x_offset, y_offset):
        """Adjust hand coordinates based on frame scaling"""
        adjusted_x = int((x * scale) + x_offset)
        adjusted_y = int((y * scale) + y_offset)
        return adjusted_x, adjusted_y

    def _create_color_boxes(self):
        boxes = {}
        box_size = max(40, self.screen_width // 30)  # Adaptive box size
        spacing = max(10, self.screen_width // 100)
        total_colors = len(self.colors)
        total_width = (box_size * total_colors) + (spacing * (total_colors - 1))
        
        x_start = (self.screen_width - total_width) // 2
        y_start = max(50, self.screen_height // 15)
        
        for i, color in enumerate(self.colors.items()):
            boxes[color[0]] = [
                (x_start + (box_size + spacing) * i, y_start),
                (x_start + box_size + (box_size + spacing) * i, y_start + box_size)
            ]
        return boxes

    def _draw_color_palette(self, frame):
        for color_name, box in self.color_boxes.items():
            cv2.rectangle(frame, box[0], box[1], self.colors[color_name], -1)
            if self.colors[color_name] == self.selected_color:
                cv2.rectangle(frame, 
                            (box[0][0] - 3, box[0][1] - 3),
                            (box[1][0] + 3, box[1][1] + 3),
                            (255, 255, 255), 2)

    def _check_color_selection(self, x, y):
        for color_name, box in self.color_boxes.items():
            if (box[0][0] < x < box[1][0] and 
                box[0][1] < y < box[1][1]):
                return self.colors[color_name]
        return None

    def _draw_return_button(self, frame):
        button_width = max(200, self.screen_width // 8)
        button_height = max(50, self.screen_height // 20)
        button_x = max(50, self.screen_width // 25)
        button_y = self.screen_height - button_height - 50
        
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), (0, 0, 255), -1)
        
        font_scale = max(0.6, self.screen_width / 2000)
        cv2.putText(frame, "Voltar menu", (button_x + 20, button_y + button_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    def _check_return_button(self, x, y):
        button_width = max(200, self.screen_width // 8)
        button_height = max(50, self.screen_height // 20)
        button_x = max(50, self.screen_width // 25)
        button_y = self.screen_height - button_height - 50
        
        return (button_x < x < button_x + button_width and 
                button_y < y < button_y + button_height)

    def _process_hands(self, frame, results, scale, x_offset, y_offset):
        if not results.multi_hand_landmarks:
            return

        original_frame_height, original_frame_width = frame.shape[:2]
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            # Original coordinates
            orig_x = int(index_tip.x * original_frame_width)
            orig_y = int(index_tip.y * original_frame_height)
            
            # Adjusted coordinates for display
            x, y = self._adjust_coordinates(orig_x, orig_y, scale, x_offset, y_offset)
            
            index_raised = index_tip.y < index_pip.y
            other_fingers_down = (
                middle_tip.y > hand_landmarks.landmark[10].y and
                ring_tip.y > hand_landmarks.landmark[14].y and
                pinky_tip.y > hand_landmarks.landmark[18].y
            )
            
            drawing_mode = index_raised and other_fingers_down
            
            if drawing_mode:
                color_box_height = max(50, self.screen_height // 15) + max(40, self.screen_width // 30)
                if y < color_box_height:
                    new_color = self._check_color_selection(x, y)
                    if new_color:
                        self.selected_color = new_color
                        self.prev_point = None
                elif self._check_return_button(x, y):
                    self.frame_count += 1
                    if self.frame_count >= 15:
                        cv2.destroyWindow('Virtual Paint')
                        return True  # Signal to return to menu
                else:
                    self.frame_count = 0
                    if self.prev_point:
                        thickness = max(3, self.screen_width // 250)
                        if handedness == "Right":
                            cv2.line(self.canvas, self.prev_point, (x, y), 
                                   self.selected_color, thickness)
                            cv2.circle(self.canvas, (x, y), 
                                     thickness//2, self.selected_color, -1)
                        else:
                            eraser_thickness = max(15, self.screen_width // 80)
                            cv2.line(self.canvas, self.prev_point, (x, y), 
                                   (0, 0, 0), eraser_thickness)
                            cv2.circle(self.canvas, (x, y), 
                                     eraser_thickness//2, (0, 0, 0), -1)
                    self.prev_point = (x, y)
            else:
                self.prev_point = None

    def run(self):
        # Initialize color boxes after screen size is set
        if not self.color_boxes:
            self.color_boxes = self._create_color_boxes()
        
        # Setup window
        self._setup_window('Virtual Paint')
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            
            # Resize frame to fit screen
            display_frame, scale, x_offset, y_offset = self._resize_frame(frame)
            
            # Resize canvas to match display frame
            if self.canvas.shape != display_frame.shape:
                self.canvas = cv2.resize(self.canvas, (display_frame.shape[1], display_frame.shape[0]))
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            self._draw_color_palette(display_frame)
            self._draw_return_button(display_frame)
            
            if results.multi_hand_landmarks:
                should_return = self._process_hands(frame, results, scale, x_offset, y_offset)
                if should_return:
                    return True
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on display frame
                    for landmark in hand_landmarks.landmark:
                        orig_x = int(landmark.x * frame.shape[1])
                        orig_y = int(landmark.y * frame.shape[0])
                        adj_x, adj_y = self._adjust_coordinates(orig_x, orig_y, scale, x_offset, y_offset)
                        cv2.circle(display_frame, (adj_x, adj_y), 3, (0, 255, 0), -1)

            display_frame = cv2.addWeighted(display_frame, 1, self.canvas, 1.0, 0)
            
            cv2.imshow('Virtual Paint', display_frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('c'):
                self.canvas = np.zeros_like(display_frame)

        return False  # Signal to exit application
