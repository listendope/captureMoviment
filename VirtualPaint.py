import cv2
import mediapipe as mp
import numpy as np

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
            return False

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
                        return True  # Signal to return to menu
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
        
        return False

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
                should_return = self._process_hands(frame, results)
                if should_return:
                    return True  # Signal to return to menu
                
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
        
        return False  # Signal to exit application