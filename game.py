"""
A game that uses blinking to launch arrows.

@author: Rhea Pandya
@version: April 2024

edited from: https://i-know-python.com/computer-vision-game-using-mediapipe-and-python/
"""

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import cv2
# import random

import time as t
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
import mediapipe as mp

import pygame
import sys

      
class Game:
    def __init__(self):
        # Load game elements
        self.score = 0

        # TODO: Initialize the Target
        # self.target = Target()
        # self.red_enemy = Enemy(RED)

        # Create the face detector
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a face landmarker instance with the video mode:
        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="data/face_landmarker.task"), running_mode=VisionRunningMode.IMAGE)
        self.landmarker = FaceLandmarker.create_from_options(options)
        # TODO: Load video
        # self.mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.VideoCapture(0))
        self.video = cv2.VideoCapture(0)

        # Get the original width and height of the image
        
        self.screen_width = 900
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def calculate_ear(self, eye_landmarks):
        # Calculate the distance between the vertical eye landmarks
        vertical_dist1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_dist2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        vertical_dist = (vertical_dist1 + vertical_dist2) / 2

        # Calculate the distance between the horizontal eye landmarks
        horizontal_dist = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        # Calculate the eye aspect ratio (EAR)
        ear = vertical_dist / (2 * horizontal_dist)

        return ear

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())
            
            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    left_eye_landmarks = np.array([(lm.x * annotated_image.shape[1], lm.y * annotated_image.shape[0]) for lm in face_landmarks[159:246]])
                    right_eye_landmarks = np.array([(lm.x * annotated_image.shape[1], lm.y * annotated_image.shape[0]) for lm in face_landmarks[386:468]])
                    left_ear = self.calculate_ear(left_eye_landmarks)
                    right_ear = self.calculate_ear(right_eye_landmarks)
                    
                    # You can adjust this threshold as needed
                    blink_threshold = .4
                    
                    if left_ear < blink_threshold and right_ear < blink_threshold:
                        print("Blink detected!")
                        self.score += 1

        return annotated_image

   
    
    def run(self):
#         """
#         Main game loop. Runs until the 
#         user presses "q".
#         """    
#         # TODO: Modify loop condition  

        image_path = "data/target.png"  # Replace "your_image_file.jpg" with the path to your image
        try:
            imageTarget = pygame.image.load(image_path)
        
            original_width = imageTarget.get_width()
            original_height = imageTarget.get_height()

            scale_factor = 0.5

            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            imageTarget = pygame.transform.scale(imageTarget, (new_width, new_height))
        except pygame.error as e:
            print("Unable to load image:", e)
            sys.exit()

        running = True
        while self.video.isOpened() and running:

#             time = False

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # the image is mirrored - flip it
            image = cv2.flip(image, 1)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            face_landmarker_result = self.landmarker.detect(to_detect)

            annotated_image = self.draw_landmarks_on_image(image, face_landmarker_result)

            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Face Tracking', annotated_image)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # Fill the screen with white color
            # screen.fill((255, 255, 255))

            # Blit the image onto the screen

            
            image_rect = imageTarget.get_rect()
            self.screen.blit(imageTarget, ((self.screen_width - new_width) // 2, (self.screen_height - new_height) // 2))

            # Update the display
            pygame.display.flip()

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break
        
        self.video.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":        
    g = Game()
    g.run()