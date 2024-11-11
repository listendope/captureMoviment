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
        
        self.color_boxes = self._create_color_boxes()
        self.selected_color = self.colors['red']
        self.brush_thickness = 5
        self.eraser_thickness = 20
        self.prev_point = None
        self.frame_count = 0

    def _create_color_boxes(self):
        boxes = {}
        box_size = 50
        spacing = 15
        total_colors = len(self.colors)
        total_width = (box_size * total_colors) + (spacing * (total_colors - 1))
        
        x_start = (1280 - total_width) // 2
        y_start = 50
        
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
        cv2.rectangle(frame, (50, 650), (250, 700), (0, 0, 255), -1)
        cv2.putText(frame, "Voltar menu", (70, 685),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _check_return_button(self, x, y):
        return 50 < x < 250 and 650 < y < 700

    def _process_hands(self, frame, results):
        if not results.multi_hand_landmarks:
            return

        frame_height, frame_width, _ = frame.shape
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            x = int(index_tip.x * frame_width)
            y = int(index_tip.y * frame_height)
            
            index_raised = index_tip.y < index_pip.y
            other_fingers_down = (
                middle_tip.y > hand_landmarks.landmark[10].y and
                ring_tip.y > hand_landmarks.landmark[14].y and
                pinky_tip.y > hand_landmarks.landmark[18].y
            )
            
            drawing_mode = index_raised and other_fingers_down
            
            if drawing_mode:
                if y < 120:
                    new_color = self._check_color_selection(x, y)
                    if new_color:
                        self.selected_color = new_color
                        self.prev_point = None
                elif self._check_return_button(x, y):
                    self.frame_count += 1
                    if self.frame_count >= 15:
                        cv2.destroyWindow('Virtual Paint')
                        menu = Menu()
                        menu.cap = self.cap
                        menu.hands = self.hands
                        menu.mp_hands = self.mp_hands
                        menu.mp_drawing = self.mp_drawing
                        menu.run()
                        return
                else:
                    self.frame_count = 0
                    if self.prev_point:
                        if handedness == "Right":
                            cv2.line(self.canvas, self.prev_point, (x, y), 
                                   self.selected_color, self.brush_thickness)
                            cv2.circle(self.canvas, (x, y), 
                                     self.brush_thickness//2, 
                                     self.selected_color, -1)
                        else:
                            cv2.line(self.canvas, self.prev_point, (x, y), 
                                   (0, 0, 0), self.eraser_thickness)
                            cv2.circle(self.canvas, (x, y), 
                                     self.eraser_thickness//2, 
                                     (0, 0, 0), -1)
                    self.prev_point = (x, y)
            else:
                self.prev_point = None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            self._draw_color_palette(frame)
            self._draw_return_button(frame)
            
            if results.multi_hand_landmarks:
                self._process_hands(frame, results)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            frame = cv2.addWeighted(frame, 1, self.canvas, 1.0, 0)
            
            cv2.imshow('Virtual Paint', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('c'):
                self.canvas = np.zeros_like(self.canvas)

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
                            menu = Menu()
                            menu.cap = self.cap
                            menu.hands = self.hands
                            menu.mp_hands = self.mp_hands
                            menu.mp_drawing = self.mp_drawing
                            menu.run()
                            return
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

class Menu:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.options = {
            'Virtual Paint': [(440, 250), (840, 350)],
            'Number Game': [(440, 400), (840, 500)]
        }
        
        self.selection_frames = 0
        self.FRAMES_TO_CONFIRM = 15

    def _draw_menu(self, frame):
        # Draw title
        title = "Escolha seu jogo"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_size = cv2.getTextSize(title, font, 2, 3)[0]
        title_x = (frame.shape[1] - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 150), font, 2, (255, 255, 255), 3)

        # Draw option boxes with enhanced visual style
        for option, coords in self.options.items():
            # Draw filled rectangle
            cv2.rectangle(frame, coords[0], coords[1], (0, 255, 0), -1)
            
            # Calculate text position for centering
            text_size = cv2.getTextSize(option, font, 1.5, 2)[0]
            text_x = coords[0][0] + (coords[1][0] - coords[0][0] - text_size[0]) // 2
            text_y = coords[0][1] + (coords[1][1] - coords[0][1] + text_size[1]) // 2
            
            # Draw text
            cv2.putText(frame, option, (text_x, text_y), font, 1.5, (255, 255, 255), 2)

    def _check_selection(self, x, y):
        for option, coords in self.options.items():
            if (coords[0][0] < x < coords[1][0] and 
                coords[0][1] < y < coords[1][1]):
                return option
        return None

    def _draw_instructions(self, frame):
        instructions = "Aponte o dedo para selecionar o jogo."
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(instructions, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, instructions, (text_x, 600), font, 1, (255, 255, 255), 2)

    def _launch_game(self, game_name):
        cv2.destroyWindow('Menu')
        if game_name == 'Virtual Paint':
            game = VirtualPaint(self.cap, self.hands, self.mp_hands, self.mp_drawing)
        else:
            game = NumberGame(self.cap, self.hands, self.mp_hands, self.mp_drawing)
        game.run()

    def run(self):
        selected_option = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            self._draw_menu(frame)
            self._draw_instructions(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * frame.shape[1])
                    y = int(index_tip.y * frame.shape[0])
                    
                    cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
                    
                    option = self._check_selection(x, y)
                    if option:
                        if option == selected_option:
                            self.selection_frames += 1
                            radius = int((self.selection_frames / self.FRAMES_TO_CONFIRM) * 20)
                            cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
                            
                            if self.selection_frames >= self.FRAMES_TO_CONFIRM:
                                self._launch_game(option)
                                return
                        else:
                            selected_option = option
                            self.selection_frames = 0
                    else:
                        selected_option = None
                        self.selection_frames = 0

            cv2.imshow('Menu', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    #x = VirtualPaint()
    #x = NumberGame()
    x = Menu()
    x.run()