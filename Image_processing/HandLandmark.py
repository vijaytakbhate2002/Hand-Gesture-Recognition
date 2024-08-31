from tensorflow import keras
import os 
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import random
import tensorflow as tf

class HandLandmark:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    custom_landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4) 
    custom_connection_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

    x_min, y_min = float('inf'), float('inf')  
    x_max, y_max = float('-inf'), float('-inf')

    landmark_colors = {
    0: (255, 0, 0),    
    1: (255, 165, 0),  # Thumb
    2: (255, 165, 0),
    3: (255, 165, 0),
    4: (255, 165, 0),
    5: (0, 255, 0),    # Index
    6: (0, 255, 0),
    7: (0, 255, 0),
    8: (0, 255, 0),
    9: (0, 255, 255),  # Middle
    10: (0, 255, 255),
    11: (0, 255, 255),
    12: (0, 255, 255),
    13: (255, 0, 255), # Ring
    14: (255, 0, 255),
    15: (255, 0, 255),
    16: (255, 0, 255),
    17: (0, 0, 255),   # Pinky
    18: (0, 0, 255),
    19: (0, 0, 255),
    20: (0, 0, 255)
    }

    def fit(self, frame:np.ndarray):
        self.frame = frame

    def co_ordinates_finder(self, results) -> tuple:
        """Args: results (process result from mediapipe model)
            Return: tuple with co-ordinates (x_min, x_max, y_min, y_max)"""
        output_image = np.zeros_like(self.frame)
        for ind, hand_landmarks in enumerate(results.multi_hand_landmarks):
            self.mp_drawing.draw_landmarks(
                image = output_image,
                landmark_list = hand_landmarks,
                connections = self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec = None,
                connection_drawing_spec = self.custom_connection_spec)
        
            x_min, y_min = float('inf'), float('inf')  
            x_max, y_max = float('-inf'), float('-inf')
            
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = self.frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(self.frame, (cx, cy), 5, self.landmark_colors.get(idx, (255, 255, 255)), -1)
                cv2.circle(output_image, (cx, cy), 5, self.landmark_colors.get(idx, (255, 255, 255)), -1)
                x_min = min(x_min, cx)
                y_min = min(y_min, cy)
                x_max = max(x_max, cx)
                y_max = max(y_max, cy)
                
            co_ordinates = (x_min, x_max, y_min, y_max)
        return output_image, co_ordinates
        
    def landmark_result(self) -> tuple:
        """Args: image
            Returns: (image, results)"""
        self.frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
        self.frame.flags.writeable = False
        results = self.hands.process(self.frame)
        self.frame.flags.writeable = True
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        return results
    
    def process_image(self) -> np.ndarray:
        results =  self.landmark_result()
        if results.multi_hand_landmarks:
            output_image, co_ordinates = self.co_ordinates_finder(results=results)
        else:
            co_ordinates = None
            output_image = self.frame
        return output_image, co_ordinates
        
    def frame_process(self) -> dict:
        """Args: frame (image)
            Return: processed_image (with landmarks) + original image + x_min, x_max, y_min, y_max """
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        output_image, co_ordinates = self.process_image()
        self.hands.close()
        return {'co_ordinates':co_ordinates, 'output_image':output_image, 'original_image':self.frame}