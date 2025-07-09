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
        
        self.score = 0
        self.current_equation = None
        self.current_answer = None
        self.frame_count = 0
        self.FRAMES_TO_CONFIRM = 15
        self.return_frame_count = 0

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
        text_size = cv2.getTextSize(text, font, size, thickness)[0]
        
        padding = 15
        cv2.rectangle(frame, 
                     (position[0] - padding, position[1] - text_size[1] - padding),
                     (position[0] + text_size[0] + padding, position[1] + padding),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text, position, font, size, color, thickness)

    def _draw_equation(self, frame):
        equation_pos = (frame.shape[1]//2 - 300, frame.shape[0]//2 + 200)
        self._draw_info_box(frame, self.current_equation, equation_pos, 3, (255, 255, 0))

    def _draw_score(self, frame):
        score_pos = (50, 80)
        self._draw_info_box(frame, f"Pontos: {self.score}", score_pos, 1.5, (0, 255, 0))

    def _draw_detected_number(self, frame, number):
        detected_pos = (frame.shape[1] - 400, 80)
        self._draw_info_box(frame, f"Dedos: {number}", detected_pos, 1.5, (0, 255, 255))

    def _draw_instructions(self, frame):
        instructions = "Mostre os dedos para completar a conta"
        inst_pos = (frame.shape[1]//2 - 350, frame.shape[0] - 50)
        self._draw_info_box(frame, instructions, inst_pos, 1, (255, 255, 255))

    def _draw_return_button(self, frame):
        cv2.rectangle(frame, (50, 650), (250, 700), (0, 0, 255), -1)
        cv2.putText(frame, "Voltar menu", (70, 685),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _check_return_button(self, x, y):
        return 50 < x < 250 and 650 < y < 700

    def run(self):
        if self.current_equation is None:
            self.current_equation, self.current_answer = self._generate_equation()
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            total_fingers = 0
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    handedness = results.multi_handedness[idx].classification[0].label
                    total_fingers += self._count_fingers(hand_landmarks, handedness)

                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * frame.shape[1])
                    y = int(index_tip.y * frame.shape[0])
                    
                    if self._check_return_button(x, y):
                        self.return_frame_count += 1
                        if self.return_frame_count >= self.FRAMES_TO_CONFIRM:
                            cv2.destroyWindow('Number Game')
                            return True  # Signal to return to menu
                    else:
                        self.return_frame_count = 0
            
            self._draw_equation(frame)
            self._draw_score(frame)
            self._draw_detected_number(frame, total_fingers)
            self._draw_instructions(frame)
            self._draw_return_button(frame)
            
            if total_fingers == self.current_answer:
                self.frame_count += 1
                if self.frame_count >= self.FRAMES_TO_CONFIRM:
                    self.score += 1
                    self.current_equation, self.current_answer = self._generate_equation()
                    self.frame_count = 0
            else:
                self.frame_count = 0
            
            cv2.imshow('Number Game', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        return False  # Signal to exit application