import cv2
import mediapipe as mp
import random
import numpy as np

# Inicializar OpenCV e Mediapipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gerar 5 números aleatórios
def generate_random_numbers():
    return [random.randint(1, 10) for _ in range(5)]

# Variáveis do jogo
target_numbers = generate_random_numbers()
current_index = 0
frame_count = 0
FRAMES_TO_CONFIRM = 15

# Variáveis para pintura
canvas = None
selected_color = None
last_point = None
colors = {
    'eraser': (0, 0, 0),
    'orange': (0, 127, 255)  # Laranja mais forte em BGR
}
color_boxes = {
    'eraser': [(580, 50), (630, 100)],    # Quadrado da borracha
    'orange': [(580, 150), (630, 200)]    # Quadrado laranja
}
click_timer = 0
last_click_time = 0

def init_canvas(frame):
    global canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

def draw_color_boxes(frame):
    for color_name, coords in color_boxes.items():
        cv2.rectangle(frame, coords[0], coords[1], colors[color_name], -1)
        if selected_color == colors[color_name]:
            cv2.rectangle(frame, 
                         (coords[0][0]-2, coords[0][1]-2), 
                         (coords[1][0]+2, coords[1][1]+2), 
                         (255, 255, 255), 2)

def check_color_selection(index_finger_tip):
    global selected_color, last_click_time, click_timer
    
    x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
    
    for color_name, coords in color_boxes.items():
        if (coords[0][0] <= x <= coords[1][0] and 
            coords[0][1] <= y <= coords[1][1]):
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - last_click_time < 0.3:  # Duplo clique detectado
                selected_color = colors[color_name]
                click_timer = 0
            last_click_time = current_time

def paint(frame, hand_landmarks):
    global last_point, canvas
    
    if selected_color is None:
        return
    
    # Contar dedos levantados
    finger_tips = [8, 12, 16, 20]
    finger_bases = [5, 9, 13, 17]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    
    thumb_up = thumb_tip.y < thumb_base.y
    
    raised_fingers = 0
    if thumb_up:
        raised_fingers += 1
        
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            raised_fingers += 1
    
    if raised_fingers == 2:
        last_point = None
        return
        
    index_tip = hand_landmarks.landmark[8]
    index_base = hand_landmarks.landmark[5]
    index_up = index_tip.y < index_base.y
    
    if index_up and thumb_up:
        x = int(index_tip.x * frame.shape[1])
        y = int(index_tip.y * frame.shape[0])
        
        if last_point is not None:
            if selected_color == colors['eraser']:
                # Usar círculo maior para apagar
                cv2.circle(canvas, (x, y), 20, (0,0,0), -1)
                if last_point:
                    # Conectar os pontos com uma linha grossa para apagar
                    cv2.line(canvas, last_point, (x, y), (0,0,0), 40)
            else:
                cv2.line(canvas, last_point, (x, y), selected_color, 2)
        last_point = (x, y)
    else:
        last_point = None

def count_fingers(hand_landmarks, handedness):
    finger_tips = [8, 12, 16, 20]
    finger_bases = [6, 10, 14, 18]
    thumb_tip = 4
    thumb_base = 2
    count = 0
    
    if handedness == 'Right':
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_base].x:
            count += 1
    else:
        if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_base].x:
            count += 1
    
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            count += 1
            
    return count

def draw_text_with_background(frame, text, position, font, scale, text_color, bg_color):
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    padding = 10
    cv2.rectangle(frame, 
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + padding),
                 bg_color,
                 -1)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    init_canvas(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb_frame)

    # Desenhar os quadrados de cores
    draw_color_boxes(frame)
    
    # Sobrepor o canvas no frame com maior opacidade
    frame = cv2.addWeighted(frame, 1, canvas, 0.8, 0)

    left_count = 0
    right_count = 0

    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            handedness = hands_results.multi_handedness[idx].classification[0].label
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Verificar seleção de cor e pintura
            check_color_selection(hand_landmarks.landmark[8])
            paint(frame, hand_landmarks)
            
            finger_count = count_fingers(hand_landmarks, handedness)
            
            if handedness == 'Left':
                left_count = finger_count
                draw_text_with_background(frame, str(finger_count), 
                                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                       2, (255, 0, 0), (255, 255, 255))
            else:
                right_count = finger_count
                draw_text_with_background(frame, str(finger_count), 
                                       (frame.shape[1] - 100, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       2, (0, 0, 255), (255, 255, 255))

        total = left_count + right_count
        center_x = frame.shape[1] // 2
        draw_text_with_background(frame, f"{total}", 
                                (center_x - 50, frame.shape[0] // 6), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                2, (0, 255, 0), (0, 0, 0))

        if total == target_numbers[current_index]:
            frame_count += 1
            if frame_count >= FRAMES_TO_CONFIRM:
                current_index = (current_index + 1) % 5
                frame_count = 0
                if current_index == 0:
                    target_numbers = generate_random_numbers()
        else:
            frame_count = 0

    # Desenhar números alvo na parte inferior
    width = frame.shape[1]
    spacing = width // 6
    for i, num in enumerate(target_numbers):
        x = spacing * (i + 1)
        y = frame.shape[0] - 30
        color = (0, 255, 0) if i == current_index else (200, 200, 200)
        bg_color = (0, 0, 0) if i == current_index else (100, 100, 100)
        draw_text_with_background(frame, str(num), (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, bg_color)

    cv2.imshow('Jogo de Soma com Dedos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
