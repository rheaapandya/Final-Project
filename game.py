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



# class Target:
#     """
#     A class to represent a random target. It spawns randomly within 
#     the given bounds.
#     """
#     def __init__(self, screen_width=600, screen_height=400):
#         self.screen_width = screen_width
#         self.screen_height = screen_height
#         # self.target_overlay()
    
#     # def respawn(self):
#     #     """
#     #     Selects a random location on the screen to respawn
#     #     """
#     #     self.x = random.randint(50, self.screen_height)
#     #     self.y = random.randint(50, self.screen_width)
    
#     # def draw(self, image):
#     #     """
#     #     Enemy is drawn as a circle onto the image

#     #     Args:
#     #         image (Image): The image to draw the enemy onto
#     #     """
#     #     cv2.circle(image, (self.x, self.y), 25, self.color, 5)

#     def target_overlay():
#         # Load the overlay image with an alpha channel (transparency)
#         target = cv2.imread('data/target.png', -1)

#         # Capture video from the webcam
#         video = cv2.VideoCapture(0)

#         while True:
#             frame = video.read()[1]

#             # Where to place the cowboy hat on the screen
#             y1, y2 = 50, 50 + target.shape[0]
#             x1, x2 = 50, 50 + target.shape[1]

#             # Saving the alpha values (transparencies)
#             alpha = target[:, :, 3] / 255.0

#             # Overlays the image onto the frame (Don't change this)
#             for c in range(0, 3):
#                 frame[y1:y2, x1:x2, c] = (alpha * target[:, :, c] +
#                                         (1.0 - alpha) * frame[y1:y2, x1:x2, c])
            
#             # Display the resulting frame
#             cv2.imshow('Target', frame)
        
      
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

        return annotated_image

    def plot_face_blendshapes_bar_graph(self, face_blendshapes):
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

        ax.set_xlabel('Score')
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        plt.show()
            
    
    def run(self):
#         """
#         Main game loop. Runs until the 
#         user presses "q".
#         """    
#         # TODO: Modify loop condition  
        while self.video.isOpened():

#             time = False

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # the image is mirrored - flip it
            image = cv2.flip(image, 1)

#             # draw the enemy on the image
#             self.green_enemy.draw(image)
#             self.red_enemy.draw(image)
#             # draw score on to screen
#             cv2.putText(image, str(self.score), (50, 50), fontFace= cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale =1, color=GREEN, thickness=2)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            face_landmarker_result = self.landmarker.detect(to_detect)

            annotated_image = self.draw_landmarks_on_image(image, face_landmarker_result)

            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Face Tracking', annotated_image)
            # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # print(face_landmarker_result)

#             # Draw the hand landmarks
#             # self.draw_landmarks_on_hand(image, results)
#             # self.check_enemy_kill(image, results)

#             # Change the color of the frame back
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Hand Tracking', image)

#             end_time = t.time()

#             # if time == True and self.score >= 10:
#             #     print(int(end_time - start_time))
#             #     break
#             # if self.inf == True and (end_time - start_time) > 2:
#             #     start_time = t.time()
#             #     self.enemy.append(Enemy(GREEN)) 
#             #     self.enemy.append(Enemy(RED))
#             #     self.enemy_count += 2
#             # if self.enemy_count >= 20:
#             #     print("Max Reached Game Over! Score: " + str(self.score))
#             #     break


            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break
        
        self.video.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":        
    g = Game()
    g.run()