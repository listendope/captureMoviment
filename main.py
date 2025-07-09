import cv2
import mediapipe as mp
from VirtualPaint import VirtualPaint
from NumberGame import NumberGame
from PeopleDetection import PeopleDetection

class Menu:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.options = {
            'Virtual Paint': [(340, 200), (640, 280)],
            'Number Game': [(340, 320), (640, 400)],
            'People Detection': [(340, 440), (640, 520)]
        }
        
        self.selection_frames = 0
        self.FRAMES_TO_CONFIRM = 15

    def _draw_menu(self, frame):
        # Draw title
        title = "Escolha seu jogo"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_size = cv2.getTextSize(title, font, 2, 3)[0]
        title_x = (frame.shape[1] - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 120), font, 2, (255, 255, 255), 3)

        # Draw option boxes with enhanced visual style
        for option, coords in self.options.items():
            # Draw filled rectangle
            cv2.rectangle(frame, coords[0], coords[1], (0, 255, 0), -1)
            
            # Calculate text position for centering
            text_size = cv2.getTextSize(option, font, 1.2, 2)[0]
            text_x = coords[0][0] + (coords[1][0] - coords[0][0] - text_size[0]) // 2
            text_y = coords[0][1] + (coords[1][1] - coords[0][1] + text_size[1]) // 2
            
            # Draw text
            cv2.putText(frame, option, (text_x, text_y), font, 1.2, (255, 255, 255), 2)

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
        elif game_name == 'Number Game':
            game = NumberGame(self.cap, self.hands, self.mp_hands, self.mp_drawing)
        elif game_name == 'People Detection':
            game = PeopleDetection(self.cap, self.hands, self.mp_hands, self.mp_drawing)
        
        return_to_menu = game.run()
        return return_to_menu

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
                                return_to_menu = self._launch_game(option)
                                if return_to_menu:
                                    # Reset selection state and continue menu loop
                                    selected_option = None
                                    self.selection_frames = 0
                                else:
                                    # Exit application
                                    self.cap.release()
                                    cv2.destroyAllWindows()
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
    menu = Menu()
    menu.run()
