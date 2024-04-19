"""
A game that uses blinking to launch arrows.

@author: Rhea Pandya
@version: April 2024

edited from: https://i-know-python.com/computer-vision-game-using-mediapipe-and-python/
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random

import time as t


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Target:
    """
    A class to represent a random target. It spawns randomly within 
    the given bounds.
    """
    def __init__(self, screen_width=600, screen_height=400):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_overlay()
    
    # def respawn(self):
    #     """
    #     Selects a random location on the screen to respawn
    #     """
    #     self.x = random.randint(50, self.screen_height)
    #     self.y = random.randint(50, self.screen_width)
    
    # def draw(self, image):
    #     """
    #     Enemy is drawn as a circle onto the image

    #     Args:
    #         image (Image): The image to draw the enemy onto
    #     """
    #     cv2.circle(image, (self.x, self.y), 25, self.color, 5)

    def target_overlay():
        # Load the overlay image with an alpha channel (transparency)
        target = cv2.imread('data/target.png', -1)

        # Capture video from the webcam
        video = cv2.VideoCapture(0)

        while True:
            frame = video.read()[1]

            # Where to place the cowboy hat on the screen
            y1, y2 = 50, 50 + target.shape[0]
            x1, x2 = 50, 50 + target.shape[1]

            # Saving the alpha values (transparencies)
            alpha = target[:, :, 3] / 255.0

            # Overlays the image onto the frame (Don't change this)
            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha * target[:, :, c] +
                                        (1.0 - alpha) * frame[y1:y2, x1:x2, c])
            
            # Display the resulting frame
            cv2.imshow('Target', frame)
        
      
class Game:
    def __init__(self):
        # Load game elements
        self.score = 0

        # TODO: Initialize the Target
        self.target = Target()
        # self.red_enemy = Enemy(RED)

        # Create the face detector
        # base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        # options = HandLandmarkerOptions(base_options=base_options,
        #                                         num_hands=2)
        # self.detector = HandLandmarker.create_from_options(options)

        # TODO: Load video
        self.video = cv2.VideoCapture(0)

    
    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())

    
    def check_enemy_intercept(self, finger_x, finger_y, enemy, image):
        """
        Determines if the finger position overlaps with the 
        enemy's position. Respawns and draws the enemy and 
        increases the score accordingly.
        Args:
            finger_x (float): x-coordinates of index finger
            finger_y (float): y-coordinates of index finger
            image (_type_): The image to draw on
        """
        if (finger_x >= enemy.x - 25) and  (finger_x <= enemy.x + 25) and (finger_y >= enemy.y - 25) and (finger_y <= enemy.y + 25):
            self.score += 1
            enemy.respawn()


    def check_enemy_kill(self, image, detection_result):
        """
        Draws a green circle on the index finger 
        and calls a method to check if we've intercepted
        with the enemy
        Args:
            image (Image): The image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get image details
        imageHeight, imageWidth = image.shape[:2]

        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # get coordinates of just the index finger

            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
            thumb = hand_landmarks[HandLandmarkPoints.THUMB_TIP.value]
            # map the coordinates back to screen dimensions
            pixelCoord = DrawingUtil._normalized_to_pixel_coordinates(finger.x,finger.y,imageWidth,imageHeight)
            pixelCoordThumb = DrawingUtil._normalized_to_pixel_coordinates(thumb.x,thumb.y,imageWidth,imageHeight)

            if pixelCoord:
                # draw the circle around the index finger
                cv2.circle(image, (pixelCoord[0], pixelCoord[1]),25,GREEN, 5)
                self.check_enemy_intercept(pixelCoord[0], pixelCoord[1], self.green_enemy, image)

            if pixelCoordThumb:
                # draw the circle around the index finger
                cv2.circle(image, (pixelCoordThumb[0], pixelCoordThumb[1]),25,RED, 5)
                self.check_enemy_intercept(pixelCoordThumb[0], pixelCoordThumb[1], self.red_enemy, image)


            
    
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # TODO: Modify loop condition  
        while self.video.isOpened():

            time = False

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # the image is mirrored - flip it
            image = cv2.flip(image, 1)

            # draw the enemy on the image
            self.green_enemy.draw(image)
            self.red_enemy.draw(image)
            # draw score on to screen
            cv2.putText(image, str(self.score), (50, 50), fontFace= cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale =1, color=GREEN, thickness=2)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks
            # self.draw_landmarks_on_hand(image, results)
            self.check_enemy_kill(image, results)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Hand Tracking', image)

            end_time = t.time()

            if time == True and self.score >= 10:
                print(int(end_time - start_time))
                break
            if self.inf == True and (end_time - start_time) > 2:
                start_time = t.time()
                self.enemy.append(Enemy(GREEN)) 
                self.enemy.append(Enemy(RED))
                self.enemy_count += 2
            if self.enemy_count >= 20:
                print("Max Reached Game Over! Score: " + str(self.score))
                break


            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break
        
        self.video.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":        
    g = Game()
    g.run()