import cv2
import mediapipe as mp
import tkinter as tk
from VirtualPaint import VirtualPaint
from NumberGame import NumberGame
from PeopleDetection import PeopleDetection

class Menu:
    def __init__(self):
        # Get screen dimensions
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        print(f"Screen resolution detected: {self.screen_width}x{self.screen_height}")
        
        self.cap = cv2.VideoCapture(0)
        # Set camera resolution to match screen aspect ratio
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Adapt menu options to screen size
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        button_width = 400
        button_height = 80
        button_spacing = 100
        
        self.options = {
            'Virtual Paint': [
                (center_x - button_width//2, center_y - button_spacing - button_height//2),
                (center_x + button_width//2, center_y - button_spacing + button_height//2)
            ],
            'Number Game': [
                (center_x - button_width//2, center_y - button_height//2),
                (center_x + button_width//2, center_y + button_height//2)
            ],
            'People Detection': [
                (center_x - button_width//2, center_y + button_spacing - button_height//2),
                (center_x + button_width//2, center_y + button_spacing + button_height//2)
            ]
        }
        
        self.selection_frames = 0
        self.FRAMES_TO_CONFIRM = 15

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

    def _draw_menu(self, frame):
        # Draw title
        title = "Escolha seu jogo"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Adapt font size to screen
        font_scale = max(1.0, self.screen_width / 1280)
        title_size = cv2.getTextSize(title, font, font_scale * 2, 3)[0]
        title_x = (frame.shape[1] - title_size[0]) // 2
        title_y = max(120, self.screen_height // 8)
        
        cv2.putText(frame, title, (title_x, title_y), font, font_scale * 2, (255, 255, 255), 3)

        # Draw option boxes
        for option, coords in self.options.items():
            # Draw filled rectangle
            cv2.rectangle(frame, coords[0], coords[1], (0, 255, 0), -1)
            
            # Calculate text position for centering
            text_size = cv2.getTextSize(option, font, font_scale * 1.2, 2)[0]
            text_x = coords[0][0] + (coords[1][0] - coords[0][0] - text_size[0]) // 2
            text_y = coords[0][1] + (coords[1][1] - coords[0][1] + text_size[1]) // 2
            
            # Draw text
            cv2.putText(frame, option, (text_x, text_y), font, font_scale * 1.2, (255, 255, 255), 2)

    def _check_selection(self, x, y):
        for option, coords in self.options.items():
            if (coords[0][0] < x < coords[1][0] and 
                coords[0][1] < y < coords[1][1]):
                return option
        return None

    def _draw_instructions(self, frame):
        instructions = "Aponte o dedo para selecionar o jogo."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, self.screen_width / 1600)
        
        text_size = cv2.getTextSize(instructions, font, font_scale, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = self.screen_height - 100
        
        cv2.putText(frame, instructions, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

    def _setup_window(self, window_name):
        """Setup window to fit screen properly"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.screen_width, self.screen_height)
        cv2.moveWindow(window_name, 0, 0)

    def _launch_game(self, game_name):
        cv2.destroyWindow('Menu')
        if game_name == 'Virtual Paint':
            game = VirtualPaint(self.cap, self.hands, self.mp_hands, self.mp_drawing)
        elif game_name == 'Number Game':
            game = NumberGame(self.cap, self.hands, self.mp_hands, self.mp_drawing)
        elif game_name == 'People Detection':
            game = PeopleDetection(self.cap, self.hands, self.mp_hands, self.mp_drawing)
        
        # Pass screen dimensions to games
        if hasattr(game, 'set_screen_size'):
            game.set_screen_size(self.screen_width, self.screen_height)
        
        return_to_menu = game.run()
        return return_to_menu

    def run(self):
        selected_option = None
        
        # Set up window
        self._setup_window('Menu')
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            
            # Resize frame to fit screen
            display_frame, scale, x_offset, y_offset = self._resize_frame(frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            self._draw_menu(display_frame)
            self._draw_instructions(display_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Adjust hand landmarks for display
                    adjusted_landmarks = []
                    for landmark in hand_landmarks.landmark:
                        adj_x, adj_y = self._adjust_coordinates(
                            landmark.x * frame.shape[1], 
                            landmark.y * frame.shape[0], 
                            scale, x_offset, y_offset
                        )
                        adjusted_landmarks.append((adj_x, adj_y))
                    
                    # Draw hand landmarks on display frame
                    for i, (x, y) in enumerate(adjusted_landmarks):
                        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                    
                    # Get index finger tip position
                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * frame.shape[1])
                    y = int(index_tip.y * frame.shape[0])
                    
                    # Adjust coordinates for screen
                    adj_x, adj_y = self._adjust_coordinates(x, y, scale, x_offset, y_offset)
                    
                    cv2.circle(display_frame, (adj_x, adj_y), 10, (255, 0, 255), -1)
                    
                    option = self._check_selection(adj_x, adj_y)
                    if option:
                        if option == selected_option:
                            self.selection_frames += 1
                            radius = int((self.selection_frames / self.FRAMES_TO_CONFIRM) * 20)
                            cv2.circle(display_frame, (adj_x, adj_y), radius, (0, 255, 255), 2)
                            
                            if self.selection_frames >= self.FRAMES_TO_CONFIRM:
                                return_to_menu = self._launch_game(option)
                                if return_to_menu:
                                    # Re-setup window after returning from game
                                    self._setup_window('Menu')
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

            cv2.imshow('Menu', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np
    menu = Menu()
    menu.run()
