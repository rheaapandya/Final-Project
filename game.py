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
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
import mediapipe as mp

import pygame
import sys

pygame.font.init()

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

        # Load arrow image
        self.arrow_img = pygame.image.load("data/arrow.png")
        self.arrow_width, self.arrow_height = self.arrow_img.get_rect().size

        # Set initial arrow position
        self.arrow_x = self.screen_width // 2 - self.arrow_width // 2
        self.arrow_y = self.screen_height // 2 - self.arrow_height // 2

        # Set initial arrow velocity
        self.arrow_vel = 5

        # Set arrow direction
        self.direction = 1  # 1 for right, -1 for left

        self.blink_detected = False

        
        self.imageTarget = pygame.image.load("data/target.png")

        original_width = self.imageTarget.get_width()
        original_height = self.imageTarget.get_height()

        scale_factor = 0.5

        self.imageTarget_width = int(original_width * scale_factor)
        self.imageTarget_height = int(original_height * scale_factor)
        self.imageTarget = pygame.transform.scale(self.imageTarget, (self.imageTarget_width, self.imageTarget_height))

        self.imageTarget_x = self.screen_width - self.imageTarget_width
        self.imageTarget_y = self.screen_height - self.imageTarget_height

        self.font = pygame.font.Font(None, 36)

        # Initialize the score
        self.score = 1000

    def draw_arrow(self):
        # Draw arrow on the screen
        self.screen.blit(self.arrow_img, (self.arrow_x, self.arrow_y))

    def update_arrow_position(self):
        if not self.blink_detected:
            self.arrow_x += self.direction * self.arrow_vel

            # Change direction if arrow reaches screen boundaries
            if self.arrow_x <= 0 or self.arrow_x >= self.screen_width - self.arrow_width:
                self.direction *= -1

    def check_arrow_position(self):
        # Check color of the pixel at arrow's position on the target image
        # print("arrow:")
        # print(self.arrow_x)
        # print("distace from middle:")
        # print(446 - self.arrow_x)

        self.score -= abs((446 - self.arrow_x)//2)


        # (self.arrow_x, self.arrow_y - ((self.arrow_height // 2) - 10))
        # point_color = self.imageTarget.get_at((self.arrow_x - self.imageTarget_x, self.arrow_y - self.imageTarget_y))
        # print(point_color)
        # Compare color of the arrow to predefined color thresholds
        # if point_color == RED:
        #     print("Arrow landed in the red region!")
        # elif point_color == YELLOW:
        #     print("Arrow landed in the yellow region!")
        # elif point_color == BLUE:
        #     print("Arrow landed in the blue region!")

    def display_score(self):
        score_text = self.font.render("Score: " + str(self.score), True, (0,0,0))
        self.screen.blit(score_text, (10, 10))

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
                        self.blink_detected = True

        return annotated_image

   
    
    def run(self):
#         """
#         Main game loop. Runs until the 
#         user presses "q".
#         """    
#         # TODO: Modify loop condition  

        
        running = True
        while self.video.isOpened() and running:

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

            self.update_arrow_position()

            self.screen.fill((255, 255, 255))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # Fill the screen with white color
            # screen.fill((255, 255, 255))

            # Blit the image onto the screen

            
            image_rect = self.imageTarget.get_rect()
            self.screen.blit(self.imageTarget, ((self.screen_width - self.imageTarget_width) // 2, (self.screen_height - self.imageTarget_height) // 2))
            self.draw_arrow()

            if self.blink_detected:
                self.check_arrow_position()
                t.sleep(5)
                self.blink_detected = False

            # Update the display

            self.display_score()
    
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