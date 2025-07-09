import cv2
import mediapipe as mp
import numpy as np
import random

class NumberGame:
    def __init__(self, cap, hands, mp_hands, mp_drawing):
        self.cap = cap
        self.hands = hands
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        
        # Default screen size (will be updated by main.py)
        self.screen_width = 1280
        self.screen_height = 720
        
        self.score = 0
        self.current_equation = None
        self.current_answer = None
        self.frame_count = 0
        self.FRAMES_TO_CONFIRM = 15
        self.return_frame_count = 0

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
        """Adjust hand coordinates based on frame scaling"""
        adjusted_x = int((x * scale) + x_offset)
        adjusted_y = int((y * scale) + y_offset)
        return adjusted_x, adjusted_y

    def _generate_equation(self):
        operations = ['+', '-']
        operation = random.choice(operations)
        
        if operation == '+':
            answer = random.randint(1, 10)
            num2 = random.randint(0, answer)
            num1 = answer - num2
        else:
            num1 = random.randint(1, 10)
            num2 = random.randint(0, num1)
            answer = num1 - num2
            
        equation = f"{num1} {operation} {num2} = ?"
        return equation, answer

    def _count_fingers(self, landmarks, handedness):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        count = 0
        
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        
        if handedness == "Right":
            if thumb_tip.x < thumb_mcp.x:
                count += 1
        else:
            if thumb_tip.x > thumb_mcp.x:
                count += 1
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
                count += 1
                
        return count

    def _draw_info_box(self, frame, text, position, size=1, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        # Adapt font size to screen
        adapted_size = size * max(0.8, self.screen_width / 1600)
        text_size = cv2.getTextSize(text, font, adapted_size, thickness)[0]
        
        padding = max(10, self.screen_width // 100)
        cv2.rectangle(frame, 
                     (position[0] - padding, position[1] - text_size[1] - padding),
                     (position[0] + text_size[0] + padding, position[1] + padding),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text, position, font, adapted_size, color, thickness)

    def _draw_equation(self, frame):
        equation_pos = (self.screen_width//2 - 200, self.screen_height//2 + 100)
        font_size = max(2.0, self.screen_width / 600)
        self._draw_info_box(frame, self.current_equation, equation_pos, font_size, (255, 255, 0))

    def _draw_score(self, frame):
        score_pos = (max(50, self.screen_width // 25), max(80, self.screen_height // 10))
        font_size = max(1.2, self.screen_width / 1000)
        self._draw_info_box(frame, f"Pontos: {self.score}", score_pos, font_size, (0, 255, 0))

    def _draw_detected_number(self, frame, number):
        detected_pos = (self.screen_width - max(300, self.screen_width // 4), max(80, self.screen_height // 10))
        font_size = max(1.2, self.screen_width / 1000)
        self._draw_info_box(frame, f"Dedos: {number}", detected_pos, font_size, (0, 255, 255))

    def _draw_instructions(self, frame):
        instructions = "Mostre os dedos para completar a conta"
        inst_pos = (self.screen_width//2 - 250, self.screen_height - max(80, self.screen_height // 10))
        font_size = max(0.8, self.screen_width / 1600)
        self._draw_info_box(frame, instructions, inst_pos, font_size, (255, 255, 255))

    def _draw_return_button(self, frame):
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
        button_width = max(200, self.screen_width // 8)
        button_height = max(50, self.screen_height // 20)
        button_x = max(50, self.screen_width // 25)
        button_y = self.screen_height - button_height - max(50, self.screen_height // 15)
        
        return (button_x < x < button_x + button_width and 
                button_y < y < button_y + button_height)

    def run(self):
        if self.current_equation is None:
            self.current_equation, self.current_answer = self._generate_equation()
        
        # Setup window
        self._setup_window('Number Game')
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Resize frame to fit screen
            display_frame, scale, x_offset, y_offset = self._resize_frame(frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            total_fingers = 0
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw hand landmarks on display frame
                    for landmark in hand_landmarks.landmark:
                        orig_x = int(landmark.x * frame.shape[1])
                        orig_y = int(landmark.y * frame.shape[0])
                        adj_x, adj_y = self._adjust_coordinates(orig_x, orig_y, scale, x_offset, y_offset)
                        cv2.circle(display_frame, (adj_x, adj_y), 3, (0, 255, 0), -1)
                    
                    handedness = results.multi_handedness[idx].classification[0].label
                    total_fingers += self._count_fingers(hand_landmarks, handedness)

                    index_tip = hand_landmarks.landmark[8]
                    orig_x = int(index_tip.x * frame.shape[1])
                    orig_y = int(index_tip.y * frame.shape[0])
                    x, y = self._adjust_coordinates(orig_x, orig_y, scale, x_offset, y_offset)
                    
                    if self._check_return_button(x, y):
                        self.return_frame_count += 1
                        if self.return_frame_count >= self.FRAMES_TO_CONFIRM:
                            cv2.destroyWindow('Number Game')
                            return True  # Signal to return to menu
                    else:
                        self.return_frame_count = 0
            
            self._draw_equation(display_frame)
            self._draw_score(display_frame)
            self._draw_detected_number(display_frame, total_fingers)
            self._draw_instructions(display_frame)
            self._draw_return_button(display_frame)
            
            if total_fingers == self.current_answer:
                self.frame_count += 1
                if self.frame_count >= self.FRAMES_TO_CONFIRM:
                    self.score += 1
                    self.current_equation, self.current_answer = self._generate_equation()
                    self.frame_count = 0
            else:
                self.frame_count = 0
            
            cv2.imshow('Number Game', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        return False  # Signal to exit application
