import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the screen dimensions
screen_width = 1000
screen_height = 950
screen = pygame.display.set_mode((screen_width, screen_height))

# Load the image
image_path = "data/target.png"  # Replace "your_image_file.jpg" with the path to your image
try:
    image = pygame.image.load(image_path)
except pygame.error as e:
    print("Unable to load image:", e)
    sys.exit()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with white color
    # screen.fill((255, 255, 255))

    # Blit the image onto the screen
    image_rect = image.get_rect()
    screen.blit(image, ((screen_width - image_rect.width) // 2, (screen_height - image_rect.height) // 2))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
