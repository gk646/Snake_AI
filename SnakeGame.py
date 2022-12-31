import sys
import os
import keras
import pygame
import time
import random
import NeuralNetwork
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


class SnakeGame:
    def __init__(self, tick_rate):
        self.game_over_variable = 0
        self.snake_movement = 1
        self.snake_speed = tick_rate
        self.highest_score1 = 0
        self.highest_score2 = 0
        pygame.font.init()
        # Window size
        self.window_x = 1000
        self.window_y = 1000

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
        self.fps = pygame.time.Clock()
        self.snake_position = [25, 25]
        self.snake_body = [[25, 25],
                           [24, 25],
                           [23, 25],
                           [22, 25]]
        self.fruit_position = [random.randrange(1, 49),
                               random.randrange(1, 49)]
        self.fruit_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.moved_fields = 0

    def rest_of_game(self):
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'
        if self.moved_fields < 250:
            self.moved_fields += 1
        else:
            self.score = 0
            self.moved_fields = 10
            self.snake_position = [51, 51]
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

    def get_game_state(self):
        screen_array = pygame.surfarray.array3d(self.surface)
        gray_array = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
        gray_array = gray_array.reshape(1, 50, 50)
        return gray_array

    def play_with_ai(self, model):
        while not self.game_over_variable:
            output = model.predict(self.get_game_state())
            action = np.argmax(output)
            if action == 0:
                self.change_to = 'UP'
            elif action == 1:
                self.change_to = 'DOWN'
            elif action == 2:
                self.change_to = 'LEFT'
            elif action == 3:
                self.change_to = 'RIGHT'
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.rest_of_game()
            if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                break
            if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                break
            for block in self.snake_body[1:]:
                if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                    break

            scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
            self.window.blit(scaled_win, (0, 0))
            pygame.display.flip()
            # Refresh game screen
            pygame.display.update()
            # Frame Per Second /Refresh Rate
            self.fps.tick(self.snake_speed)

    def reset_game(self):
        self.snake_position = [25, 25]
        self.snake_body = [[25, 25],
                           [24, 25],
                           [23, 25],
                           [22, 25]]
        self.fruit_position = [random.randrange(1, 49),
                               random.randrange(1, 49)]
        self.fruit_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.moved_fields = 0

    def train_models(self, crossover_1, crossover_2):
        scores_plot = []
        c = 0
        self.highest_score2 = 0
        self.highest_score1 = 0
        for b in range(15):
            scores = []
            weights = []
            crossover_1, crossover_2 = NeuralNetwork.cross_over(crossover_1, crossover_2)
            for i in range(25):
                half_random = NeuralNetwork.half_random_model(crossover_1)
                print("First Loop - " + str(i))
                self.reset_game()
                while True:
                    output = half_random.predict(self.get_game_state(), verbose=0)
                    action = np.argmax(output)
                    if action == 0:
                        self.change_to = 'UP'
                    elif action == 1:
                        self.change_to = 'DOWN'
                    elif action == 2:
                        self.change_to = 'LEFT'
                    elif action == 3:
                        self.change_to = 'RIGHT'
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    self.rest_of_game()
                    if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                        break
                    if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                        break

                    # Touching the snake body
                    for block in self.snake_body[1:]:
                        if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                            break

                    scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
                    self.window.blit(scaled_win, (0, 0))
                    pygame.display.flip()
                    # Refresh game screen
                    pygame.display.update()
                    # Frame Per Second /Refresh Rate
                    self.fps.tick(self.snake_speed)
                score = self.score * 1000 + self.moved_fields
                weights.append(half_random.get_weights())
                scores.append(score)
            scores = np.array(scores)
            scores_plot.append([scores.mean(), c])
            c += 1
            print(scores.mean())
            if np.max(scores) > self.highest_score1:
                crossover_1 = NeuralNetwork.build_model()
                crossover_1.set_weights(weights[scores.argmax()])
                self.highest_score1 = np.max(scores)
            scores = []
            weights = []
            for i in range(25):
                half_random2 = NeuralNetwork.half_random_model(crossover_2)
                print("Second Loop - " + str(i))
                self.reset_game()
                while True:
                    output = half_random2.predict(self.get_game_state(), verbose=0)
                    action = np.argmax(output)
                    if action == 0:
                        self.change_to = 'UP'
                    elif action == 1:
                        self.change_to = 'DOWN'
                    elif action == 2:
                        self.change_to = 'LEFT'
                    elif action == 3:
                        self.change_to = 'RIGHT'
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    self.rest_of_game()
                    if self.snake_position[0] < 0 or self.snake_position[0] > 50:
                        break
                    if self.snake_position[1] < 0 or self.snake_position[1] > 50:
                        break

                    # Touching the snake body
                    for block in self.snake_body[1:]:
                        if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                            break

                    scaled_win = pygame.transform.scale(self.surface, self.window.get_size())
                    self.window.blit(scaled_win, (0, 0))
                    pygame.display.flip()
                    # Refresh game screen
                    pygame.display.update()
                    # Frame Per Second /Refresh Rate
                    self.fps.tick(self.snake_speed)
                score = self.score * 1000 + self.moved_fields
                weights.append(half_random2.get_weights())
                scores.append(score)
            scores = np.array(scores)
            scores_plot.append([scores.mean(), c])
            c += 1
            print(scores.mean())
            if np.max(scores) > self.highest_score2:
                crossover_2 = NeuralNetwork.build_model()
                crossover_2.set_weights(weights[scores.argmax()])
                self.highest_score1 = np.max(scores)

        best_model, best_model2 = NeuralNetwork.cross_over(crossover_1, crossover_2)
        best_model.save('genetic1')
        best_model2.save('genetic2')
        iterations, scores = zip(*scores_plot)
        plt.plot(scores, iterations)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Scores over Iterations')
        plt.show()


game = SnakeGame(200)
snakeNet1 = keras.models.load_model('genetic1')
snakeNet2 = keras.models.load_model('genetic2')
game.train_models(snakeNet1, snakeNet2)

# snakeNet.save("snakeModel")
