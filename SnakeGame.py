import sys

import pygame
import time
import random
import NeuralNetwork
import numpy as np
import cv2


class SnakeGame:
    def __init__(self, tickrate):

        self.snake_movement = 1
        self.snake_speed = tickrate
        pygame.font.init()
        # Window size
        self.window_x = 500
        self.window_y = 500

        self.window = pygame.display.set_mode((self.window_x, self.window_y))
        # defining colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)
        pygame.display.set_caption('snake')

        self.surface = pygame.Surface((50, 50))

        self.scaled_surface = pygame.transform.scale(self.surface, (200, 200))
        # FPS (frames per second) controller
        self.fps = pygame.time.Clock()

        # defining snake default position
        self.snake_position = [25, 25]

        # defining first 4 blocks of snake body
        self.snake_body = [[25, 25],
                           [24, 25],
                           [23, 25],
                           [22, 25]]
        # fruit position
        self.fruit_position = [random.randrange(1, 49),
                               random.randrange(1, 49)]

        self.fruit_spawn = True

        # setting default snake direction towards
        # right
        self.direction = 'RIGHT'
        self.change_to = self.direction

        # initial score
        self.score = 0

    def show_score(self, color, font, size):
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)

        # create the display surface object
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, color)

        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()

        # displaying text
        self.surface.blit(score_surface, score_rect)

    def game_over(self):
        # creating font object my_font
        my_font = pygame.font.SysFont('times new roman', 50)

        # creating a text surface on which text
        # will be drawn
        game_over_surface = my_font.render(
            'Your Score is : ' + str(self.score), True, self.red)

        # create a rectangular object for the text
        # surface object
        game_over_rect = game_over_surface.get_rect()

        # setting position of the text
        game_over_rect.midtop = (self.window_x / 2, self.window_y / 4)

        # blit will draw the text on screen
        self.surface.blit(game_over_surface, game_over_rect)
        pygame.display.flip()

        # after 2 seconds we will quit the program
        time.sleep(2)

        # deactivating pygame library
        pygame.quit()

        # quit the program
        quit()

    def play(self):
        # Main Function

        while True:
            print(self.get_game_state().shape)
            # handling key events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.change_to = 'UP'
                    if event.key == pygame.K_DOWN:
                        self.change_to = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        self.change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        self.change_to = 'RIGHT'
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # If two keys pressed simultaneously
            # we don't want snake to move into two
            # directions simultaneously
            if self.change_to == 'UP' and self.direction != 'DOWN':
                self.direction = 'UP'
            if self.change_to == 'DOWN' and self.direction != 'UP':
                self.direction = 'DOWN'
            if self.change_to == 'LEFT' and self.direction != 'RIGHT':
                self.direction = 'LEFT'
            if self.change_to == 'RIGHT' and self.direction != 'LEFT':
                self.direction = 'RIGHT'

            # Moving the snake
            if self.direction == 'UP':
                self.snake_position[1] -= self.snake_movement
            if self.direction == 'DOWN':
                self.snake_position[1] += self.snake_movement
            if self.direction == 'LEFT':
                self.snake_position[0] -= self.snake_movement
            if self.direction == 'RIGHT':
                self.snake_position[0] += self.snake_movement

            # Snake body growing mechanism
            # if fruits and snakes collide then scores
            # will be incremented by 10
            self.snake_body.insert(0, list(self.snake_position))
            if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
                self.score += 1
                self.fruit_spawn = False
            else:
                self.snake_body.pop()

            if not self.fruit_spawn:
                self.fruit_position = self.fruit_position = [random.randrange(1, 50),
                                                             random.randrange(1, 50)]

            self.fruit_spawn = True
            self.surface.fill(self.black)

            for pos in self.snake_body:
                pygame.draw.rect(self.surface, self.green,
                                 pygame.Rect(pos[0], pos[1], 1, 1))
                pygame.draw.rect(self.surface, self.white, pygame.Rect(
                    self.fruit_position[0], self.fruit_position[1], 1, 1))

            # Game Over conditions
            if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                self.game_over()
            if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                self.game_over()

            # Touching the snake body
            for block in self.snake_body[1:]:
                if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                    self.game_over()

            # displaying score countinuously
            self.show_score(self.white, 'times new roman', 2)

            scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
            self.window.blit(scaled_win, (0, 0))
            pygame.display.flip()
            # Refresh game screen
            pygame.display.update()
            # Frame Per Second /Refresh Rate
            self.fps.tick(self.snake_speed)

    def play_with_ai(self, model):
        # Main Function

        while True:
            # Get the output of the model
            output = model.predict(self.get_game_state())

            # Determine the action to take
            action = np.argmax(output)

            # Map the action to a key input
            if action == 0:
                # Move up
                self.change_to = 'UP'
            elif action == 1:
                # Move down
                self.change_to = 'DOWN'
            elif action == 2:
                # Move left
                self.change_to = 'LEFT'
            elif action == 3:
                # Move right
                self.change_to = 'RIGHT'
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if self.change_to == 'UP' and self.direction != 'DOWN':
                self.direction = 'UP'
            if self.change_to == 'DOWN' and self.direction != 'UP':
                self.direction = 'DOWN'
            if self.change_to == 'LEFT' and self.direction != 'RIGHT':
                self.direction = 'LEFT'
            if self.change_to == 'RIGHT' and self.direction != 'LEFT':
                self.direction = 'RIGHT'

                # Moving the snake
            if self.direction == 'UP':
                self.snake_position[1] -= self.snake_movement
            if self.direction == 'DOWN':
                self.snake_position[1] += self.snake_movement
            if self.direction == 'LEFT':
                self.snake_position[0] -= self.snake_movement
            if self.direction == 'RIGHT':
                self.snake_position[0] += self.snake_movement

            self.snake_body.insert(0, list(self.snake_position))
            if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
                self.score += 1
                self.fruit_spawn = False
            else:
                self.snake_body.pop()

            if not self.fruit_spawn:
                self.fruit_position = self.fruit_position = [random.randrange(1, 50),
                                                             random.randrange(1, 50)]

            self.fruit_spawn = True
            self.surface.fill(self.black)

            for pos in self.snake_body:
                pygame.draw.rect(self.surface, self.green,
                                 pygame.Rect(pos[0], pos[1], 1, 1))
                pygame.draw.rect(self.surface, self.white, pygame.Rect(
                    self.fruit_position[0], self.fruit_position[1], 1, 1))

            # Game Over conditions
            if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                self.game_over()
            if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                self.game_over()

            # Touching the snake body
            for block in self.snake_body[1:]:
                if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                    self.game_over()

            # displaying score countinuously
            self.show_score(self.white, 'times new roman', 2)

            scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
            self.window.blit(scaled_win, (0, 0))
            pygame.display.flip()
            # Refresh game screen
            pygame.display.update()
            # Frame Per Second /Refresh Rate
            self.fps.tick(self.snake_speed)

    def get_game_state(self):
        screen_array = pygame.surfarray.array3d(self.surface)
        gray_array = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
        gray_array =  gray_array.reshape(1,50,50)
        return gray_array


game = SnakeGame(10)
snakeNet = NeuralNetwork.build_model()
game.play_with_ai(snakeNet)
